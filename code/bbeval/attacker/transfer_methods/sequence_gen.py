"""
    Dataset (part) for this available at: https://github.com/Joey-la/MalSeqData
"""
import torch.nn as nn
import torch as ch
import torch.nn.functional as F
from typing import Tuple

from bbeval.config.core import ExperimentConfig
from bbeval.models.core import GenericModelWrapper
from bbeval.config import AttackerConfig
from bbeval.attacker.core import Attacker


class HeaderEvasion(Attacker):
    def __init__(self,
                 model: GenericModelWrapper,
                 aux_models: dict,
                 config: AttackerConfig,
                 experiment_config: ExperimentConfig):
        super().__init__(model, aux_models, config, experiment_config)
        self._init_params()
        self._define_attacker()
    
    def _init_params(self):
        self.iterations = 50
        self.random_init = False
        self.optimize_all_dos = False
        self.threshold = 0.5

        # 128 dim Word2Vec, PIP loss
        # Adam, 1e-4 for sub, 1e-1 for gen
        # 0.2 dropout, <=50 epochs
        # BS 128
    
    def _define_attacker(self):
        # Create embedding
        self.embedding = nn.Embedding(self.input_vocab_size, self.emb_dim)

        # Extract one-hot representations of API callnames from data
        # Convert to word embedding vectors

        # Provide sequence as input to LSTM
        # Used to generate L sequence outputs for every k input words
        self.gen_lstm = GenerativeLSTM(
            self.embedding,
            output_dim = self.input_vocab_size,
        )
        # Proxy for victim malware classifier
        self.sub_bilstm = DiscriminativeLSTM(
            self.embedding)

    def _attack(self, x_orig, x_adv, y_label, y_target=None):
        # Start training loop untin convergence
        num_queries_used = 0
        for i in range(self.iterations):
            # TODO: Split input sequences into k-size chunks
            # Get outputs from self.gen_lstm for each chunk and iterweave
            # Pass modified output to self.sub_bilstm
            # Get output logit
            # Compute and back-prop loss_c, update BiLSTM
            # Compute and back-prop loss_g, update LSTM
            # loss_g = p log y_hat
            # loss_c = CE
            # Check for convergence
            num_queries_used += len(x_adv)
            pass


class Encoder(nn.Module):
    def __init__(self,
                 embedding,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float,
                 bidirectional: bool = False):
        super().__init__()

        self.embedding = embedding
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional = False)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: ch.Tensor) -> Tuple[ch.Tensor]:

        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = ch.tanh(self.fc(ch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: ch.Tensor,
                encoder_outputs: ch.Tensor) -> ch.Tensor:

        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = ch.tanh(self.attn(ch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = ch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 embedding,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.embedding = embedding
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: ch.Tensor,
                              encoder_outputs: ch.Tensor) -> ch.Tensor:
        # Apply attentuin
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = ch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: ch.Tensor,
                decoder_hidden: ch.Tensor,
                encoder_outputs: ch.Tensor) -> Tuple[ch.Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = ch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(ch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class GenerativeLSTM(nn.Module):
    def __init__(self,
                 embedding,
                 output_dim: int,
                 emb_dim: int = 128,
                 enc_hid_dim: int = 128,
                 dec_hid_dim: int = 128,
                 attn_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        # input_dim -> number of tokens, essentially
        self.embedding = embedding
        # Create encoder
        self.encoder = Encoder(
                embedding=self.embedding,
                emb_dim=emb_dim,
                enc_hid_dim=enc_hid_dim,
                dec_hid_dim=dec_hid_dim,
                dropout=dropout)
        # Create attention
        self.attention = Attention(
                enc_hid_dim=enc_hid_dim,
                dec_hid_dim=dec_hid_dim,
                attn_dim=attn_dim)
        # Create decoder
        self.decoder = Decoder(
            embedding=self.embedding,
            output_dim=output_dim,
            emb_dim=emb_dim,
            enc_hid_dim=enc_hid_dim,
            dec_hid_dim=dec_hid_dim,
            dropout=dropout,
            attention=self.attention)

    def forward(self,
                src: ch.Tensor,
                trg: ch.Tensor,
                num_outputs: int) -> ch.Tensor:
        # Take note of batch size and target vocabulary
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Placeholder to store predictions
        outputs = ch.zeros(num_outputs, batch_size, trg_vocab_size, device=src.device)

        # Get encoder outputs
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        # Generate output tokens
        for t in range(1, num_outputs):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            # Gumbel-Softmax based predictions
            output = F.gumbel_softmax(output, tau=1, hard=False)
            outputs[t] = output

        return outputs


class DiscriminativeLSTM(nn.Module):
    def __init__(self,
                 embedding,
                 emb_dim: int = 128,
                 enc_hid_dim: int = 128,
                 dec_hid_dim: int = 128,
                 attn_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        # input_dim -> number of tokens, essentially
        self.embedding = embedding
        # Create encoder
        self.encoder = Encoder(
                embedding=self.embedding,
                emb_dim=emb_dim,
                enc_hid_dim=enc_hid_dim,
                dec_hid_dim=dec_hid_dim,
                dropout=dropout,
                bidirectional=True)
        # Create attention
        self.attention = Attention(
                enc_hid_dim=enc_hid_dim,
                dec_hid_dim=dec_hid_dim,
                attn_dim=attn_dim)
        self.fc = nn.Linear(enc_hid_dim * 2, 1)

    def forward(self, src: ch.Tensor) -> ch.Tensor:

        # batch_size = src.shape[1]
        # trg_vocab_size = self.decoder.output_dim

        # outputs = ch.zeros(num_outputs, batch_size, trg_vocab_size, device=src.device)

        # Get encoder outputs
        encoder_outputs, _ = self.encoder(src)

        # Get logits
        return self.fc(encoder_outputs[-1])
