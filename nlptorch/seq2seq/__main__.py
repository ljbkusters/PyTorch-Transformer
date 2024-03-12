"""Sequence to sequence model

German to English translator

LSTM based network for language translation


```

| -------------- Encoder -------------------- |    | ----------- Decoder ----------- |

                                                        Good*     Morning** <EOS>
                                                         ^         ^         ^
                                                         |         |         |
h0 -> [LSTM] -> [LSTM] -> [LSTM] -> [LSTM] -> vec-z -> [LSTM] -> [LSTM] -> [LSTM]
          ^       ^         ^         ^                  ^         ^         ^
          |       |         |         |                  |         |         |
        <SOS>   Guten    Morgen     <EOS>              <SOS>     Good*     Morning**
```
"""
import argparse
import collections
import logging
import typing

import torch
import torch.utils.tensorboard
import torch.utils.data
import torchtext.datasets  # dataset to train on
# from torchtext.legacy.data import Field, BucketIterator
import tqdm

from nlptorch.tokenization import INIT_TOKEN, END_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN
from nlptorch import tokenization
from nlptorch import seq2seq
from nlptorch import checkpointing


LOGGER = logging.getLogger(__name__)


def build_vocab(data: typing.Iterable[str],
                tokenizer: typing.Callable
                ) -> torchtext.vocab.Vocab:
    """Create a vocab from a dataset"""
    counter = collections.Counter()
    for string in data:
        counter.update(tokenizer(string))
    vocab = torchtext.vocab.vocab(counter,
                                  specials=(INIT_TOKEN,
                                            END_TOKEN,
                                            PAD_TOKEN,
                                            UNKNOWN_TOKEN)
                                  )
    vocab.set_default_index(vocab[UNKNOWN_TOKEN])
    return vocab


def _load_model(args: argparse.Namespace, logger: logging.Logger,
                input_vocab: torchtext.vocab.Vocab,
                output_vocab: torchtext.vocab.Vocab) -> seq2seq.Seq2Seq:
    logger.info("Initializing new model")
    encoder = seq2seq.S2SEncoder(vocab_input_size=len(input_vocab),
                                 embedding_size=args.encoder_embedding_size,
                                 hidden_size=args.hidden_size,
                                 num_layers=args.num_layers,
                                 p_dropout=args.encoder_dropout)
    decoder = seq2seq.S2SDecoder(vocab_input_size=len(input_vocab),
                                 vocab_output_size=len(output_vocab),
                                 embedding_size=args.decoder_embedding_size,
                                 hidden_size=args.hidden_size,
                                 num_layers=args.num_layers,
                                 p_dropout=args.decoder_dropout,
                                 )
    model = seq2seq.Seq2Seq(encoder=encoder,
                            decoder=decoder,
                            input_vocab=input_vocab,
                            output_vocab=output_vocab,
                            p_target_force_ratio=0.5)
    if args.checkpoint is not None:
        logger.warning("loading models is currently not supported!")
    return model


def _get_device(args: argparse, logger: logging.Logger) -> torch.device:
    device_str = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    return torch.device(device_str)


def _get_raw_data(args: argparse.Namespace, logger: logging.Logger = LOGGER
                  ) -> tuple[list[tuple[str]]]:
    logger.debug("getting raw data...")
    dsets: tuple = torchtext.datasets.Multi30k(
        root=".data",
        split=('train', 'valid'),
        language_pair=(args.input_language, args.output_language)
        )
    logger.debug(dsets)
    train_dset = list(dsets[0])
    valid_dset = list(dsets[1])
    return train_dset, valid_dset


def _get_tokenizers(args: argparse.Namespace,
                    logger: logging.Logger = LOGGER
                    ) -> tuple[typing.Callable]:
    input_tokenizer = torchtext.data.utils.get_tokenizer(
        'spacy', language=args.input_language
        )
    output_tokenizer = torchtext.data.utils.get_tokenizer(
        'spacy', language=args.output_language
        )
    return input_tokenizer, output_tokenizer


def _get_vocabs(args: argparse.Namespace,
                input_tokenizer: typing.Callable,
                output_tokenizer: typing.Callable,
                train_data: typing.Iterable,
                logger: logging.Logger = LOGGER
                ) -> tuple[torchtext.vocab.Vocab]:
    logger.debug("getting SpaCy tokenizers vocabs...")
    logger.debug("generating vocabs tokenizers vocabs...")
    input_vocab = build_vocab(tokenization.input_generator(train_data), input_tokenizer)
    output_vocab = build_vocab(tokenization.output_generator(train_data), output_tokenizer)
    return input_vocab, output_vocab


def _get_dataloaders(args: argparse.Namespace,
                     input_vocab: torchtext.vocab.Vocab,
                     output_vocab: torchtext.vocab.Vocab,
                     train_data: typing.Iterable,
                     valid_data: typing.Iterable,
                     logger: logging.Logger = LOGGER,
                     ) -> torch.utils.data.DataLoader:
    dl_options = dict(batch_size=args.batch_size,
                      shuffle=args.shuffle_batches,
                      collate_fn=tokenization.get_add_special_token_collate_fn(
                          init_idx=input_vocab[INIT_TOKEN],
                          end_idx=input_vocab[END_TOKEN],
                          pad_idx=input_vocab[PAD_TOKEN],
                          ),
                      )
    train_dl = torch.utils.data.DataLoader(train_data, **dl_options)
    valid_dl = torch.utils.data.DataLoader(valid_data, **dl_options)
    return train_dl, valid_dl


def tokenize_data(input_vocab: torchtext.vocab.Vocab,
                  output_vocab: torchtext.vocab.Vocab,
                  input_tokenizer: typing.Callable,
                  output_tokenizer: typing.Callable,
                  dataset: typing.Iterable[tuple[str]]):
    data = []
    for (input_item, output_item) in dataset:
        input_tensor = torch.Tensor([input_vocab[token]
                                     for token in input_tokenizer(input_item)]).long()
        output_tensor = torch.Tensor([output_vocab[token]
                                      for token in output_tokenizer(output_item)]).long()
        data.append((input_tensor, output_tensor))
    return data


def main(args: argparse.Namespace, logger: logging.Logger = LOGGER):
    """Trains a german to english translator"""
    # initialize data
    train_data, valid_data = _get_raw_data(args)
    input_tokenizer, output_tokenizer = _get_tokenizers(args)
    input_vocab, output_vocab = _get_vocabs(args, input_tokenizer, output_tokenizer, train_data)
    train_data = tokenize_data(input_vocab, output_vocab,
                               input_tokenizer, output_tokenizer,
                               train_data)
    valid_data = tokenize_data(input_vocab, output_vocab,
                               input_tokenizer, output_tokenizer,
                               valid_data)
    train_dl, valid_dl = _get_dataloaders(args, input_vocab, output_vocab,
                                          train_data, valid_data)

    # initialize model
    model: seq2seq.Seq2Seq = _load_model(args, logger,
                                         input_vocab=input_vocab,
                                         output_vocab=output_vocab)
    device: torch.device = _get_device(args, logger)
    model.to(device)

    # initialize tensorboard
    tb_summary_writer: torch.utils.tensorboard.SummaryWriter = \
        torch.utils.tensorboard.SummaryWriter("runs/loss_plot")
    tb_step: int = 0

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=input_vocab[PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_running_average = collections.deque([], maxlen=100)

    for epoch in range(args.num_epochs):
        logger.info(f'Epoch [{epoch+1}/{args.num_epochs}]')
        tqdm_loader = tqdm.tqdm(enumerate(train_dl))
        for batch_idx, (input, target) in tqdm_loader:
            input = input.to(device)
            logger.debug(f"input shape {input.shape}")
            logger.debug(f"target shape {target.shape}")
            target = target.to(device)

            output = model(input, target)

            # check if this is still relevant...
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = loss_fn(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            tb_summary_writer.add_scalar("Training loss", loss, global_step=tb_step)
            tb_step += 1

            loss_running_average.append(loss.detach().item())
            tqdm_loader.set_postfix({"loss": sum(loss_running_average)/len(loss_running_average)})
        checkpointing.save_checkpoint(model, ".model_checkpoints/s2s_checkpoint.pth.tar")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    train_hp: argparse._ArgumentGroup = parser.add_argument_group("training_hyperparams")
    train_hp.add_argument("--num-epochs", type=int, default=20)
    train_hp.add_argument("--learning-rate", type=float, default=1e-3)
    train_hp.add_argument("--batch-size", type=int, default=64)
    # TODO expand choices based on dataset
    train_hp.add_argument("--input-language", type=str, default="de",
                          choices=("de", "en"))
    train_hp.add_argument("--output-language", type=str, default="en",
                          choices=("de", "en"))
    train_hp.add_argument("--shuffle-batches", type=bool, default=True)

    model_hp: argparse._ArgumentGroup = parser.add_argument_group("model_hyperparams")
    model_hp.add_argument("--checkpoint", type=str, default=None)
    model_hp.add_argument("--cpu", action="store_true", help="use the CPU even if GPU is available")
    model_hp.add_argument("--encoder-embedding-size", type=int, default=300)
    model_hp.add_argument("--decoder-embedding-size", type=int, default=300)
    model_hp.add_argument("--hidden-size", type=int, default=1024, help="Size of hidden vector")
    model_hp.add_argument("--num-layers", type=int, default=2, help="number of LSTM layers")
    model_hp.add_argument("--encoder-dropout", type=float, default=0.5,
                          help="Dropout percentage in encoder")
    model_hp.add_argument("--decoder-dropout", type=float, default=0.5,
                          help="Dropout percentage in decoder")

    # tensorboard_args: argparse._ArgumentGroup = parser.add_argument_group("tensorboard")
    # tensorboard_args.add_argument("")

    logging_args = parser.add_argument_group("logging_args")
    logging_args.add_argument("--log-level", type=str, default="INFO")

    args: argparse.Namespace = parser.parse_args()

    # configure logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(
        level=numeric_level,
        format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.input_language == args.output_language:
        raise RuntimeError("Input language equals output language!")

    main(args)
