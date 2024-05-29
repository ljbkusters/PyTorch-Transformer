import argparse
import collections
import logging
import typing
import torch
import torch.utils.tensorboard
import torch.utils.data
import torchtext.datasets
import tqdm
from torchtext.vocab import Vocab

from nlptorch.transformer._modules import Transformer, TransformerWrapper
from nlptorch.tokenization import INIT_TOKEN, END_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN
from nlptorch import tokenization
from nlptorch import checkpointing

LOGGER = logging.getLogger(__name__)


def build_vocab(data: typing.Iterable[str], tokenizer: typing.Callable) -> Vocab:
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


def _get_device(args: argparse.Namespace, logger: logging.Logger) -> torch.device:
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
    input_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language=args.input_language)
    output_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language=args.output_language)
    return input_tokenizer, output_tokenizer


def _get_vocabs(args: argparse.Namespace,
                input_tokenizer: typing.Callable,
                output_tokenizer: typing.Callable,
                train_data: typing.Iterable,
                logger: logging.Logger = LOGGER
                ) -> tuple[Vocab]:
    logger.debug("getting SpaCy tokenizers vocabs...")
    logger.debug("generating vocabs tokenizers vocabs...")
    input_vocab = build_vocab(tokenization.input_generator(train_data), input_tokenizer)
    output_vocab = build_vocab(tokenization.output_generator(train_data), output_tokenizer)
    return input_vocab, output_vocab


def _get_dataloaders(args: argparse.Namespace,
                     input_vocab: Vocab,
                     output_vocab: Vocab,
                     train_data: typing.Iterable,
                     valid_data: typing.Iterable,
                     logger: logging.Logger = LOGGER) -> torch.utils.data.DataLoader:
    dl_options = dict(batch_size=args.batch_size,
                      shuffle=args.shuffle_batches,
                      collate_fn=tokenization.get_add_special_token_collate_fn(
                          init_idx=input_vocab[INIT_TOKEN],
                          end_idx=input_vocab[END_TOKEN],
                          pad_idx=input_vocab[PAD_TOKEN],
                          )
                      )
    train_dl = torch.utils.data.DataLoader(train_data, **dl_options, num_workers=4)
    valid_dl = torch.utils.data.DataLoader(valid_data, **dl_options, num_workers=4)
    return train_dl, valid_dl


def tokenize_data(input_vocab: Vocab,
                  output_vocab: Vocab,
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


def _get_fresh_model(args: argparse.Namespace,
                     logger: logging.Logger,
                     input_vocab: Vocab,
                     output_vocab: Vocab) -> TransformerWrapper:
    logger.info("Initializing new model")
    transformer_model = Transformer(
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        num_reps=args.num_reps,
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        dropout=args.dropout
    )
    model = TransformerWrapper(
        transformer=transformer_model,
        input_vocab=input_vocab,
        output_vocab=output_vocab,
        deterministic=True
    )
    return model


# Main training function
def main(args: argparse.Namespace, logger: logging.Logger = LOGGER):
    if args.input_language == args.output_language:
        raise RuntimeError("Input language equals output language!")
    train_data, valid_data = _get_raw_data(args)
    input_tokenizer, output_tokenizer = _get_tokenizers(args)
    input_vocab, output_vocab = _get_vocabs(args, input_tokenizer, output_tokenizer, train_data)
    train_data = tokenize_data(input_vocab, output_vocab,
                               input_tokenizer, output_tokenizer,
                               train_data)
    valid_data = tokenize_data(input_vocab, output_vocab,
                               input_tokenizer, output_tokenizer,
                               valid_data)
    train_dl, valid_dl = _get_dataloaders(args, input_vocab, output_vocab, train_data, valid_data)

    # Initialize model
    if args.checkpoint is None:
        model = _get_fresh_model(args, logger, input_vocab, output_vocab)
    else:
        model = checkpointing.load_checkpoint_with_config(args.checkpoint)
    device = _get_device(args, logger)
    model.to(device)

    # Initialize tensorboard
    tb_summary_writer = torch.utils.tensorboard.SummaryWriter("runs/loss_plot")
    tb_step = 0

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=input_vocab[PAD_TOKEN])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    loss_running_average = collections.deque([], maxlen=100)
    test_sentence = "Ein Mädchen läuft entlang die Straße."

    for epoch in range(args.num_epochs):
        logger.info(f'Epoch [{epoch+1}/{args.num_epochs}]')
        logger.info(f'test sentence: {test_sentence}')
        model.eval()
        # logger.info(f'model output: {model.translate_sentence(test_sentence, input_tokenizer)}')
        model.train()
        tqdm_loader = tqdm.tqdm(train_dl)
        for batch_idx, (input, target) in enumerate(tqdm_loader):
            input = input.to(device)
            target = target.to(device)
            output = model(input, target[:, :-1])  # Pass the input and target with <EOS> token removed at the end

            # Flatten output for computing loss: ignore the first token (<SOS>)
            # in the target for loss computation
            output_flat = output.view(-1, model.transformer.output_vocab_size)
            target_flat = target[:, 1:].contiguous().view(-1)  # Shape: (batch_size * target_length)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(output_flat, target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            tb_summary_writer.add_scalar("Training loss", loss, global_step=tb_step)
            tb_step += 1

            loss_running_average.append(loss.item())
            tqdm_loader.set_postfix({"loss": sum(loss_running_average) / len(loss_running_average)})

        # checkpointing.save_checkpoint_with_config(model, config, ".model_checkpoints/s2s_checkpoint.pth.tar") # Update this for TransformerWrapper

# Define argument parser and add arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="mode")

logging_args = parser.add_argument_group("Logging arguments")
logging_args.add_argument("--log-level", type=str, default="INFO")

train_parser = subparsers.add_parser("train", help="Train the model")
train_parser.set_defaults(func=main)
train_hp = train_parser.add_argument_group("Training hyperparameters")
train_hp.add_argument("--num-epochs", type=int, default=20)
train_hp.add_argument("--learning-rate", type=float, default=1e-3)
train_hp.add_argument("--batch-size", type=int, default=64)
train_hp.add_argument("--input-language", type=str, default="de", choices=("de", "en"))
train_hp.add_argument("--output-language", type=str, default="en", choices=("de", "en"))
train_hp.add_argument("--shuffle-batches", type=bool, default=True)

model_hp = train_parser.add_argument_group("Model hyperparameters")
model_hp.add_argument("--checkpoint", type=str, default=None)
model_hp.add_argument("--cpu", action="store_true", help="use the CPU even if GPU is available")
model_hp.add_argument("--embedding-size", type=int, default=300)
model_hp.add_argument("--num-heads", type=int, default=8)
model_hp.add_argument("--num-reps", type=int, default=6)
model_hp.add_argument("--dropout", type=float, default=0.1)

eval_parser = subparsers.add_parser("eval", help="Run the model in eval mode")
eval_parser.set_defaults(func=None) # Define a separate eval function for TransformerWrapper if needed
eval_parser.add_argument("--checkpoint", type=str, help="path to checkpoint file")
eval_parser.add_argument("--input", dest="input", help="Input sentence to translate")

args = parser.parse_args()

numeric_level = getattr(logging, args.log_level.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {args.log_level}')
logging.basicConfig(level=numeric_level, format='[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

if not hasattr(args, 'func'):
    parser.print_help()
    exit(1)
args.func(args, LOGGER)
