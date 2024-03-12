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

Seq2Seq with Attention

```
The idea of attention:
As a human, when we are translating a sentence we will likely
pay attention to different words in the input space for given words in the output space.

We also want to apply this to our neural network. Attention here means calculating a set of
weights which weigh various factors of the input space and/or hidden states. In Seq2Seq with
attention these attention weights are calculated using a neural network `a`.

Specifically the neural network `a` calculates the energy values e_ij
```latex
    e_{ij} = a(s_{i-1}, h_j)
```
the $e_ij$ are then softmaxed into attention weights a_ij (this way attention is normalized)

The attention network therefore learns what to pay attention to given a specific input token and
a current hidden state

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

import torch
import torchtext

from nlptorch import tokenization


LOGGER = logging.getLogger(__name__)


class S2SEncoder(torch.nn.Module):
    def __init__(self,
                 vocab_input_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 p_dropout: float = 0.,
                 bidirectional: bool = True,
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
                                 dropout=p_dropout,
                                 bidirectional=bidirectional)

    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        # x shape (num_input_words, batch_size)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell


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

        next_decoder_input_word = self.SOS_TOKEN

        if self.training:
            # training mode
            next_decoder_input_word = target[0]
            target_len: int = target.shape[0]
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
            while next_decoder_input_word != self.EOS_TOKEN:
                output, hidden, cell = self.decoder(next_decoder_input_word, hidden, cell)
                outputs[t_idx] = output
                best_guess_idx: int = torch.argmax(output, dim=1)
                next_decoder_input_word = self.output_vocab[best_guess_idx]
            return torch.stack(outputs)
