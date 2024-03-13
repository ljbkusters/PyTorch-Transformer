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
import logging
import typing
import random
import dataclasses

import torch
import torchtext

from nlptorch import tokenization


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class S2SEncoderConfig():
    vocab_input_size: int
    embedding_size: int
    hidden_size: int
    num_layers: int
    p_dropout: int


class S2SEncoder(torch.nn.Module):
    def __init__(self,
                 vocab_input_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 p_dropout: float = 0.
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = torch.nn.Dropout(p_dropout)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_input_size,
                                            embedding_dim=embedding_size)
        self.rnn = torch.nn.LSTM(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 dropout=p_dropout)

    @classmethod
    def from_config(cls, config: S2SEncoderConfig):
        return cls(vocab_input_size=config.vocab_input_size,
                   embedding_size=config.embedding_size,
                   hidden_size=config.hidden_size,
                   num_layers=config.num_layers,
                   p_dropout=config.p_dropout)

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        # x shape (num_input_words, batch_size)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell


@dataclasses.dataclass
class S2SDecoderConfig():
    vocab_input_size: int
    vocab_output_size: int
    embedding_size: int
    hidden_size: int
    num_layers: int
    p_dropout: float


class S2SDecoder(torch.nn.Module):

    def __init__(self,
                 vocab_input_size: int,
                 vocab_output_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 p_dropout: float = 0.
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        LOGGER.debug(f"decoder setup::vocab_input_size: {vocab_input_size}")
        LOGGER.debug(f"decoder setup::hidden_size: {hidden_size}")
        LOGGER.debug(f"decoder setup::num_layers: {num_layers}")
        LOGGER.debug(f"decoder setup::input_size: {embedding_size}")

        self.dropout = torch.nn.Dropout(p_dropout)
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_input_size,
            embedding_dim=embedding_size)
        self.rnn = torch.nn.LSTM(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 dropout=p_dropout)
        self.fc = torch.nn.Linear(hidden_size, vocab_output_size)

    @classmethod
    def from_config(cls, config: S2SDecoderConfig):
        return cls(vocab_input_size=config.vocab_input_size,
                   vocab_output_size=config.vocab_output_size,
                   embedding_size=config.embedding_size,
                   hidden_size=config.hidden_size,
                   num_layers=config.num_layers,
                   p_dropout=config.p_dropout
                   )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        LOGGER.debug(f"decoder forward::x: {x.shape}")
        # shape of x (batch_size) -> (num_words = 1, batch_size)
        single_word_batch_x = torch.unsqueeze(x, 0)

        # embedding shape (num_words = 1, batch_size, embedding dim)
        embedding = self.dropout(self.embedding(single_word_batch_x))

        # shape of outputs (num_words = 1, batch_size, hidden_size)
        LOGGER.debug(f"decoder forward::embedding: {embedding.shape}")
        LOGGER.debug(f"decoder forward::hidden: {hidden.shape}")
        LOGGER.debug(f"decoder forward::cell: {cell.shape}")
        outputs, (hidden, cell) = self.rnn(input=embedding, hx=(hidden, cell))

        # shape of predicitons (num_words = 1, batch_size, length_of_vocab)
        predictions = self.fc(outputs)
        return predictions.squeeze(0), hidden, cell


@dataclasses.dataclass
class Seq2SeqConfig():
    encoder_config: S2SEncoderConfig
    decoder_config: S2SDecoderConfig
    input_vocab: torchtext.vocab.Vocab
    output_vocab: torchtext.vocab.Vocab
    p_train_target_force_ratio: float


class Seq2Seq(torch.nn.Module):

    SOS_TOKEN: str = tokenization.INIT_TOKEN  # start of signal
    EOS_TOKEN: str = tokenization.END_TOKEN   # end of signal

    def __init__(self,
                 encoder: S2SEncoder,
                 decoder: S2SDecoder,
                 input_vocab: torchtext.vocab.Vocab,
                 output_vocab: torchtext.vocab.Vocab,
                 p_target_force_ratio: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.__p_train_target_force_ratio = p_target_force_ratio

    @classmethod
    def from_config(cls, config: Seq2SeqConfig):
        encoder = S2SEncoder.from_config(config.encoder_config)
        decoder = S2SDecoder.from_config(config.decoder_config)
        return cls(encoder=encoder,
                   decoder=decoder,
                   input_vocab=config.input_vocab,
                   output_vocab=config.output_vocab,
                   p_target_force_ratio=config.p_train_target_force_ratio,
                   )

    def translate_sentence(self, input: str,
                           input_tokenizer: typing.Callable,
                           max_len: int = 50):
        LOGGER.debug(input)
        input_list = [self.input_vocab[token] for token in input_tokenizer(input)]
        LOGGER.debug(input_list)
        if len(input_list) > max_len:
            LOGGER.error(f"Length of input too long! ({len(input_list)=:} > {max_len=:})")
            return None
        input_tensor = torch.Tensor(input_list).long().unsqueeze(1)
        LOGGER.debug(input_tensor)
        output_tensor = self.forward(input_tensor).squeeze()
        output = self.output_vocab.lookup_tokens(output_tensor.tolist())
        return output

    def forward(self,
                source: typing.List[str],
                target: typing.Optional[typing.List[str]] = None,
                ):
        """
        Parameters
        ----------
        - source (List[str]): Input string
        - target (Optional[List[str]]): Target translation
        """
        if (self.training) and (target is None):
            raise RuntimeError("target vector was ommited which is required in in training mode!")

        _, batch_size = source.shape
        target_vocab_length = len(self.output_vocab)
        hidden, cell = self.encoder(source)

        next_decoder_input_word = torch.as_tensor(self.output_vocab[self.SOS_TOKEN]
                                                  ).long().unsqueeze(0)

        if self.training:
            # training mode
            next_decoder_input_word = target[0]
            target_len: int = target.shape[0]
            LOGGER.debug(f"first decoder word {next_decoder_input_word}")
            LOGGER.debug(next_decoder_input_word.shape)
            outputs: torch.Tensor = torch.zeros(target_len, batch_size, target_vocab_length
                                                ).to(next(self.parameters()).device)
            for t_idx in range(1, target_len):
                output, hidden, cell = self.decoder(next_decoder_input_word, hidden, cell)
                outputs[t_idx] = output
                best_guess_idx: torch.LongTensor = torch.argmax(output, dim=1)
                if random.random() < self.__p_train_target_force_ratio:
                    next_decoder_input_word = target[t_idx]
                else:
                    next_decoder_input_word = best_guess_idx
            return outputs
        else:
            # inference mode
            outputs = []
            outputs.append(next_decoder_input_word)
            while (next_decoder_input_word.item() != self.output_vocab[self.EOS_TOKEN]
                   and len(outputs) < 50):
                LOGGER.debug(f"first decoder word {next_decoder_input_word}")
                LOGGER.debug(next_decoder_input_word.shape)
                output, hidden, cell = self.decoder(next_decoder_input_word, hidden, cell)
                best_guess_idx: int = torch.argmax(output, dim=1)
                next_decoder_input_word = best_guess_idx
                outputs.append(next_decoder_input_word)
            return torch.stack(outputs)
