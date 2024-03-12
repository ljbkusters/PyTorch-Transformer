import functools
import logging
import typing

import torch

LOGGER = logging.getLogger(__name__)

UNKNOWN_TOKEN: str = "<unk>"
INIT_TOKEN: str = "<sos>"
END_TOKEN: str = "<eos>"
PAD_TOKEN: str = "<pad>"
SPECIAL_TOKENS = (UNKNOWN_TOKEN, INIT_TOKEN, END_TOKEN, PAD_TOKEN)


def input_generator(data_generator: typing.Iterable) -> typing.Generator:
    """Expects the data_generator to be a generator of tuples of the form
    (input sentence (str), translated sentence (str))
    """
    for example in data_generator:
        yield example[0]


def output_generator(data_generator: typing.Iterable) -> typing.Generator:
    """Expects the data_generator to be a generator of tuples of the form
    (input sentence (str), translated sentence (str))
    """
    for example in data_generator:
        yield example[1]


def add_init_and_end_token_to_item(data_item: torch.Tensor, init_idx, end_idx):
    return torch.cat((
        torch.tensor([init_idx]),
        data_item,
        torch.tensor([end_idx]),
    ))


def get_add_special_token_collate_fn(init_idx: int, end_idx: int, pad_idx: int
                                     ) -> functools.partialmethod:
    """Returns the partial of _add_special_tokens_to_batch for given token indices"""
    return functools.partial(_add_special_tokens_to_batch,
                             init_idx=init_idx,
                             end_idx=end_idx,
                             pad_idx=pad_idx,
                             )


def _add_special_tokens_to_batch(data_batch: typing.Iterable[tuple],
                                 init_idx: int,
                                 end_idx: int,
                                 pad_idx: int):
    """Can be used as DataLoader collate_fn to prepare a batch of raw data

    Prepends INIT_TOKEN, appends END_TOKEN and pads batch with PAD_TOKEN

    Parameters
    ----------
    - data_batch: an iterable data batch created by the dataloader
    - init_idx: int

    Returns
    -------
        processed batch
    """
    input_batch, output_batch = [], []
    for (input_item, output_item) in data_batch:
        input_batch.append(add_init_and_end_token_to_item(input_item, init_idx, end_idx))
        output_batch.append(add_init_and_end_token_to_item(output_item, init_idx, end_idx))
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, padding_value=pad_idx)
    output_batch = torch.nn.utils.rnn.pad_sequence(output_batch, padding_value=pad_idx)
    return input_batch, output_batch
