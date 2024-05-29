import typing
import dataclasses

import torch
import torchtext

from nlptorch import tokenization


def _create_triangular_mask(batch_size: int, mask_size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(mask_size, mask_size))
    return mask.unsqueeze(0).expand(batch_size, -1, -1)


class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, keys_dim: int):
        super().__init__()
        self.keys_dim = keys_dim
        self.one_over_sqrt_keys_dim = 1 / torch.sqrt(torch.tensor(keys_dim))

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask: typing.Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """ScaledDotProductAttention

        Calculates `attn(Q, K, V) = matmul(softmax(matmul(Q, K.T)/sqrt(d_k)), V)`

        Args:
            queries (torch.Tensor): queries vector tensor of shape (batch_size, sequence_length, d_q)
            keys (torch.Tensor): keys vector tensor of shape (batch_size, sequence_length, d_k)
            values (torch.Tensor): keys vector tensor of shape (batch_size, sequence_length, d_v)
            mask (torch.Tensor): mask tensor of shape (batch_size, sequence_length, sequence_length)

        Returns:
            torch.Tensor: output vector of shape (batch_size, d_v)
        """
        energy = torch.matmul(queries, torch.transpose(keys, 1, 2))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        scaled_energy = energy * self.one_over_sqrt_keys_dim
        attention = torch.softmax(scaled_energy, dim=-1)
        valued_attention = torch.matmul(attention, values)
        return valued_attention


class MultiHeadAttention(torch.nn.Module):

    def __init__(self,
                 num_heads: int,
                 embedding_dim: int,
                 queries_dim: typing.Optional[int] = None,
                 keys_dim: typing.Optional[int] = None,
                 values_dim: typing.Optional[int] = None,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.queries_dim = queries_dim if queries_dim is not None else embedding_dim // num_heads
        self.keys_dim = keys_dim if keys_dim is not None else embedding_dim // num_heads
        self.values_dim = values_dim if values_dim is not None else embedding_dim // num_heads

        self.proj_queries = torch.nn.Linear(self.embedding_dim, self.num_heads * self.queries_dim)
        self.proj_keys = torch.nn.Linear(self.embedding_dim, self.num_heads * self.keys_dim)
        self.proj_values = torch.nn.Linear(self.embedding_dim, self.num_heads * self.values_dim)

        self.proj_output = torch.nn.Linear(self.values_dim * self.num_heads, self.embedding_dim)

        self.heads = tuple(ScaledDotProductAttention(self.keys_dim) for _ in range(num_heads))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask: typing.Optional[torch.Tensor] = None):
        multi_head_queries = torch.reshape(self.proj_queries(queries),
                                           (*queries.shape[:2], self.queries_dim, self.num_heads))
        multi_head_keys = torch.reshape(self.proj_keys(keys),
                                        (*keys.shape[:2], self.keys_dim, self.num_heads))
        multi_head_values = torch.reshape(self.proj_values(values),
                                          (*values.shape[:2], self.values_dim, self.num_heads))
        outputs = tuple(head(multi_head_queries[..., i],
                             multi_head_keys[..., i],
                             multi_head_values[..., i],
                             mask
                             )
                        for i, head in enumerate(self.heads)
                        )
        concatenated_output = torch.concat(outputs, dim=-1)
        return self.dropout(self.proj_output(concatenated_output))


class TransformerFeedForward(torch.nn.Module):

    def __init__(self, embedding_dim: int, inner_dim: int,
                 inner_dropout: float = 0.,
                 outer_dropout: float = 0.,
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.inner_dim = inner_dim
        self.inner_dropout = torch.nn.Dropout(inner_dropout)
        self.outer_dropout = torch.nn.Dropout(outer_dropout)
        self.fc1 = torch.nn.Linear(self.embedding_dim, self.inner_dim)
        self.fc2 = torch.nn.Linear(self.inner_dim, self.embedding_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.outer_dropout(self.fc2(
            self.inner_dropout(self.relu(self.fc1(embedding)))
            ))


class PositionalEncoding(torch.nn.Module):

    BASE = torch.tensor((10_000,))

    def __init__(self, embedding_size: int, max_seq_len: int = 5000,
                 embedding_scale_factor: float = 1.):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.embedding_scale_factor = embedding_scale_factor

        positional_encoding = torch.zeros((max_seq_len, embedding_size))
        seq_positions = torch.arange(max_seq_len).unsqueeze(1)

        frequency_term = torch.exp( -1 * torch.log(self.BASE)
                                   * torch.arange(0, embedding_size, 2)
                                   / embedding_size)
        positional_encoding[:, 0::2] = torch.sin(seq_positions * frequency_term)
        positional_encoding[:, 1::2] = torch.cos(seq_positions * frequency_term)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, embedding):
        encoded_embedding = (embedding * self.embedding_scale_factor
                             + self.positional_encoding[:embedding.size(1), :].unsqueeze(0))
        return encoded_embedding


class TransformerEncoderBlock(torch.nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads,
                                                       embedding_dim=embedding_dim)
        self.feed_forward = TransformerFeedForward(embedding_dim=embedding_dim,
                                                   inner_dim=4*embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
        attn_emedding = self.multi_head_attention(input_embedding,
                                                  input_embedding,
                                                  input_embedding)
        x = self.layer_norm(input_embedding + attn_emedding)
        x = self.layer_norm(x + self.feed_forward(x))
        return self.dropout(x)


class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int = 8,
                 masked_mha_dropout: float = 0.,
                 mha_dropout: float = 0.,
                 block_dropout: float = 0.,
                 ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.masked_multi_head_attention = MultiHeadAttention(num_heads=num_heads,
                                                              embedding_dim=embedding_dim,
                                                              dropout=masked_mha_dropout)
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads,
                                                       embedding_dim=embedding_dim,
                                                       dropout=mha_dropout,
                                                       )
        self.block_dropout = torch.nn.Dropout(block_dropout)
        self.feed_forward = TransformerFeedForward(embedding_dim=embedding_dim,
                                                   inner_dim=4 * embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self,
                encoder_embedding: torch.Tensor,
                output_embedding: torch.Tensor,
                mask: typing.Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        x = self.layer_norm(
            output_embedding
            + self.masked_multi_head_attention(output_embedding,
                                               output_embedding,
                                               output_embedding,
                                               mask))
        x = self.layer_norm(x + self.multi_head_attention(x, encoder_embedding, encoder_embedding))
        x = self.layer_norm(x + self.feed_forward(x))
        return x


class OutputProbabilityProjection(torch.nn.Module):

    def __init__(self, embedding_size: int, vocab_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.linear = torch.nn.Linear(embedding_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        b, s, e = embedding.shape
        return self.softmax(self.linear(embedding[:, -1, :].squeeze()))


class Transformer(torch.nn.Module):

    def __init__(self, embedding_size: int, num_heads: int, num_reps: int,
                 input_vocab_size: int,
                 output_vocab_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_reps = num_reps
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.dropout = dropout
        self.memory_cache = None
        self.input_embedding = torch.nn.Embedding(input_vocab_size, embedding_size)
        self.output_embedding = torch.nn.Embedding(output_vocab_size, embedding_size)
        self.input_positional_encoding = PositionalEncoding(embedding_size)
        self.output_positional_encoding = PositionalEncoding(embedding_size)
        self.encoder_stack_sequential: torch.nn.Sequential = torch.nn.Sequential(*list(
            TransformerEncoderBlock(self.embedding_size, self.num_heads)
            for _ in range(num_reps)))
        self.decoder_stack: tuple[TransformerDecoderBlock] = tuple(
            TransformerDecoderBlock(self.embedding_size, self.num_heads)
            for _ in range(num_reps))
        self.output_probability_projection = OutputProbabilityProjection(
            embedding_size=self.embedding_size,
            vocab_size=self.output_vocab_size)

    def reset_memory(self):
        self.memory_cache = None

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        # by saving the input encoding to a cache we prevent recomputation
        if self.memory_cache is None:
            encoded_input_embedding = self.input_positional_encoding(self.input_embedding(inputs))
            self.memory_cache = self.encoder_stack_sequential(encoded_input_embedding)

        # decoder computation can probably be made more efficient as well!
        # We do not have to recompute the output embedding again for the
        # previous n-1 output tokens, but this model doesn't take that into account
        x = self.output_positional_encoding(self.output_embedding(outputs))
        b, s, _ = x.shape
        mask = _create_triangular_mask(batch_size=b, mask_size=s)
        for decoder_block in self.decoder_stack:
            x = decoder_block(self.memory_cache, x, mask)
        return self.output_probability_projection(x)


class TransformerWrapper(torch.nn.Module):
    """Wraps the transformer to do sequential decoding with right shifting"""

    INIT_TOKEN = tokenization.INIT_TOKEN
    END_TOKEN = tokenization.END_TOKEN

    def __init__(self, transformer: Transformer,
                 input_vocab: torchtext.vocab.Vocab,
                 output_vocab: torchtext.vocab.Vocab,
                 deterministic: bool = True,
                 p_force_target_training: float = 0.5,
                 max_seq_size: int = 50,
                 ):
        super().__init__()
        self.transformer = transformer
        self.deterministic = deterministic

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.p_force_target_training = p_force_target_training
        self.max_seq_size = max_seq_size

    def forward(self,
                encoder_sequence: torch.Tensor,
                target_sequence: torch.Optional[torch.Tensor] = None
                ):
        self.transformer.reset_memory()
        if self.training and target_sequence is None:
            raise RuntimeError("When training a target_sequence is required but None was provided!")

        if self.training:
            return self.__training_forward(encoder_sequence, target_sequence)
        else:
            return self.__eval_forward(encoder_sequence)

    def __training_forward(self,
                           encoder_sequence: torch.Tensor,
                           target_sequence: torch.Tensor,
                           ):
        # forward untill target seq length, used in train mode
        outputs = [target_sequence[:, 0]]
        batch_size = target_sequence.size(0)
        for t_idx in range(1, target_sequence.shape[1]):
            output_probabilities = self.transformer(encoder_sequence,
                                                    torch.stack(outputs, dim=1))
            force_target_mask = torch.rand(batch_size) < self.p_force_target_training
            next_token = torch.zeros(batch_size, dtype=int, device=target_sequence.device)
            next_token[force_target_mask] = target_sequence[force_target_mask, t_idx]
            next_token[~force_target_mask] = \
                torch.argmax(output_probabilities, dim=-1)[~force_target_mask]
            outputs.append(next_token)
        return outputs

    def __eval_forward(self,
                       encoder_sequence: torch.Tensor,
                       ):
        # forward untill EOS or max_seq_len reached, used in eval mode
        batch_size = encoder_sequence.size(0)
        # Initialize with the batched INIT_TOKEN
        outputs = torch.full((batch_size, 1), self.output_vocab[self.INIT_TOKEN],
                             device=encoder_sequence.device)
        # Create a mask to keep track of sequences that have not yet reached EOS
        unfinished_mask = torch.ones(batch_size, dtype=torch.bool, device=encoder_sequence.device)
        # Lengths for each sequence to know when to stop
        lengths = torch.zeros(batch_size, dtype=torch.long, device=encoder_sequence.device)

        while True:
            last_probabilities = self.transformer(encoder_sequence, outputs)
            if self.deterministic:
                next_word = torch.argmax(last_probabilities, dim=-1, keepdim=True)
            else:
                next_word = torch.multinomial(last_probabilities, 1)

            # Mask for sequences where the next word is not EOS and length limit is not reached
            unfinished_mask &= (next_word.squeeze(dim=-1) != self.output_vocab[self.END_TOKEN])
            unfinished_mask &= (lengths < self.max_seq_size)
            lengths += unfinished_mask

            # Stack new tokens to the outputs
            outputs = torch.cat([outputs, next_word], dim=-1)

            if not unfinished_mask.any():
                break

        # Remove sequences that went beyond max_seq_size
        outputs = [output[:length] for output, length in zip(outputs, lengths)]
        return outputs


if __name__ == "__main__":
    BTCH_SIZE = 32
    SEQ_LEN = 50
    EMB_DIM = 512
    embedding = torch.rand(BTCH_SIZE, SEQ_LEN, EMB_DIM)
    print(f"{embedding.shape=}")

    def count_parameters(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    mha = MultiHeadAttention(num_heads=8, embedding_dim=EMB_DIM)
    out = mha(embedding, embedding, embedding)
    print(f"{out.shape=}")

    trans_enc = TransformerEncoderBlock(EMB_DIM, num_heads=8, dropout=0.1)
    trans_dec = TransformerDecoderBlock(EMB_DIM, num_heads=8,
                                        mha_dropout=0.1,
                                        masked_mha_dropout=0.1)
    out = trans_enc.forward(embedding)
    print(f"enc: {out.shape=}")
    mask = _create_triangular_mask(BTCH_SIZE, SEQ_LEN//2)
    out = trans_dec.forward(out, embedding[:, ::2, ...])
    print(f"dec: {out.shape=}")

    pos_enc = PositionalEncoding(EMB_DIM)
    pos_enc(embedding)
    print(f"{out.shape=}")

    vocab_size = 10_000
    input_sequence = torch.randint(0, vocab_size, (BTCH_SIZE, SEQ_LEN))
    output_sequence = torch.randint(0, vocab_size, (BTCH_SIZE, 3))
    print(f"{input_sequence.shape=}")
    print(f"{output_sequence.shape=}")
    trf = Transformer(embedding_size=EMB_DIM, num_heads=8, num_reps=6,
                      input_vocab_size=10_000,
                      output_vocab_size=10_000,
                      )
    print(count_parameters(trf))
    trf_out = trf.forward(input_sequence, output_sequence)
    print(f"{trf_out.shape=}")

    @dataclasses.dataclass
    class MockVocab():
        size: int

        def __len__(self):
            return self.size

        def __getitem__(self, key):
            return 0

    trf_wrapped = TransformerWrapper(trf, MockVocab(vocab_size), MockVocab(vocab_size),
                                     deterministic=True)
    out = trf_wrapped(input_sequence, output_sequence)
    print(f"wrapped train: {len(out)=}")
    trf_wrapped.eval()
    out = trf_wrapped(input_sequence)
    print(f"wrapped eval: {[bi.shape for bi in out]=}")
