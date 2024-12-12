import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Time2Vec(nn.Module):
    def __init__(self, activation, batch_size, in_len, in_features, hidden_dim=32):
        super(Time2Vec, self).__init__()
        if activation == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos
        self.out_features = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, 2)
        self.w0 = nn.parameter.Parameter(torch.randn(batch_size, in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(batch_size, in_len, 1))
        self.w = nn.parameter.Parameter(torch.randn(batch_size, in_features, self.out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(batch_size, in_len, self.out_features - 1))

    def forward(self, x):
        v1 = self.activation(torch.matmul(x, self.w) + self.b)
        v2 = torch.matmul(x, self.w0) + self.b0
        v3 = torch.cat([v1, v2], -1)
        x = self.fc1(v3)
        return x


class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class MSCNN(nn.Module):
    def __init__(self, in_features, out_features=32, se=16, dropout=0.2):
        super(MSCNN, self).__init__()
        self.kernel_size = [3, 7, 11]
        self.dilation = [1, 2]
        self.chomp11 = Chomp1d((self.kernel_size[0]-1) * self.dilation[0])
        self.conv11 = nn.Conv1d(in_features, out_features, kernel_size=self.kernel_size[0], dilation=self.dilation[0], stride=1, padding=(self.kernel_size[0]-1) * self.dilation[0])
        self.relu11 = nn.ReLU()
        self.dropout11 = nn.Dropout(dropout)
        self.chomp12 = Chomp1d((self.kernel_size[0] - 1) * self.dilation[1])
        self.conv12 = nn.Conv1d(out_features, out_features, kernel_size=self.kernel_size[0], dilation=self.dilation[1],
                               stride=1, padding=(self.kernel_size[0] - 1) * self.dilation[1])
        self.relu12 = nn.ReLU()
        self.dropout12 = nn.Dropout(dropout)

        self.net1 = nn.Sequential(self.conv11, self.chomp11, self.relu11, self.dropout11,
                                  self.conv12, self.chomp12, self.relu12, self.dropout12)

        self.chomp21 = Chomp1d((self.kernel_size[1] - 1) * self.dilation[0])
        self.conv21 = nn.Conv1d(in_features, out_features, kernel_size=self.kernel_size[1], dilation=self.dilation[0],
                                stride=1, padding=(self.kernel_size[1] - 1) * self.dilation[0])
        self.relu21 = nn.ReLU()
        self.dropout21 = nn.Dropout(dropout)
        self.chomp22 = Chomp1d((self.kernel_size[1] - 1) * self.dilation[1])
        self.conv22 = nn.Conv1d(out_features, out_features, kernel_size=self.kernel_size[1], dilation=self.dilation[1],
                                stride=1, padding=(self.kernel_size[1] - 1) * self.dilation[1])
        self.relu22 = nn.ReLU()
        self.dropout22 = nn.Dropout(dropout)

        self.net2 = nn.Sequential(self.conv21, self.chomp21, self.relu21, self.dropout21,
                                  self.conv22, self.chomp22, self.relu22, self.dropout22)

        self.chomp31 = Chomp1d((self.kernel_size[2] - 1) * self.dilation[0])
        self.conv31 = nn.Conv1d(in_features, out_features, kernel_size=self.kernel_size[2], dilation=self.dilation[0],
                                stride=1, padding=(self.kernel_size[2] - 1) * self.dilation[0])
        self.relu31 = nn.ReLU()
        self.dropout31 = nn.Dropout(dropout)
        self.chomp32 = Chomp1d((self.kernel_size[2] - 1) * self.dilation[1])
        self.conv32 = nn.Conv1d(out_features, out_features, kernel_size=self.kernel_size[2], dilation=self.dilation[1],
                                stride=1, padding=(self.kernel_size[2] - 1) * self.dilation[1])
        self.relu32 = nn.ReLU()
        self.dropout32 = nn.Dropout(dropout)

        self.net3 = nn.Sequential(self.conv31, self.chomp31, self.relu31, self.dropout31,
                                  self.conv32, self.chomp32, self.relu32, self.dropout32)

        self.SELayer = SELayer1D(out_features * 3, se)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.net1(x)
        x2 = self.net2(x)
        x3 = self.net3(x)
        x = torch.cat([x1, x2, x3], dim=-2)
        x = self.SELayer(x)
        x = x.transpose(1, 2)

        return x


class MSCformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, batch_size, seq_len, label_len, out_len,
                 t2v_hidden=32, factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, device=torch.device('cuda:0')):
        super(MSCformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_ele_t2v = Time2Vec('sin', batch_size, seq_len, enc_in, t2v_hidden)
        self.enc_cool_t2v = Time2Vec('sin', batch_size, seq_len, enc_in, t2v_hidden)
        self.enc_heat_t2v = Time2Vec('sin', batch_size, seq_len, enc_in, t2v_hidden)

        self.dec_t2v = Time2Vec('sin', batch_size, label_len + out_len, dec_in, t2v_hidden)

        self.channel = 16

        self.enc_ele_cnn = MSCNN(enc_in + 2, self.channel)
        self.enc_cool_cnn = MSCNN(enc_in + 2, self.channel)
        self.enc_heat_cnn = MSCNN(enc_in + 2, self.channel)

        self.dec_cnn = MSCNN(dec_in + 2, self.channel)

        self.enc_ele_down = nn.Conv1d(enc_in, 3 * self.channel, 1)
        self.enc_cool_down = nn.Conv1d(enc_in, 3 * self.channel, 1)
        self.enc_heat_down = nn.Conv1d(enc_in, 3 * self.channel, 1)

        self.enc_selayer = SELayer1D(3 * 3 * self.channel)

        self.dec_down = nn.Conv1d(dec_in, 3 * self.channel, 1)

        eemb_in = 3 * 3 * self.channel
        demb_in = 3 * self.channel

        # Encoding
        self.enc_embedding = DataEmbedding(eemb_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(demb_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection1 = nn.Linear(d_model, c_out, bias=True)
        self.projection2 = nn.Linear(d_model, c_out, bias=True)
        self.projection3 = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x_ele_enc = x_enc
        x_cool_enc = x_enc
        x_heat_enc = x_enc

        res_ele_enc = self.enc_ele_down(x_ele_enc.transpose(1, 2)).transpose(1, 2)
        x_ele_enc_t2v = self.enc_ele_t2v(x_ele_enc)
        x_ele_enc = torch.cat([x_ele_enc, x_ele_enc_t2v], dim=-1)
        x_ele_enc = res_ele_enc + self.enc_ele_cnn(x_ele_enc)

        res_cool_enc = self.enc_cool_down(x_cool_enc.transpose(1, 2)).transpose(1, 2)
        x_cool_enc_t2v = self.enc_cool_t2v(x_cool_enc)
        x_cool_enc = torch.cat([x_cool_enc, x_cool_enc_t2v], dim=-1)
        x_cool_enc = res_cool_enc + self.enc_cool_cnn(x_cool_enc)

        res_heat_enc = self.enc_heat_down(x_heat_enc.transpose(1, 2)).transpose(1, 2)
        x_heat_enc_t2v = self.enc_heat_t2v(x_heat_enc)
        x_heat_enc = torch.cat([x_heat_enc, x_heat_enc_t2v], dim=-1)
        x_heat_enc = res_heat_enc + self.enc_heat_cnn(x_heat_enc)

        x_enc_sum = torch.cat([x_ele_enc, x_cool_enc, x_heat_enc], dim=-1)

        x_enc_sum = x_enc_sum.permute(0, 2, 1)
        x_enc_sum = self.enc_selayer(x_enc_sum).transpose(1, 2)

        enc_out = self.enc_embedding(x_enc_sum, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        x_dec = x_dec[:, :, -3:]
        res_dec = self.dec_down(x_dec.transpose(1, 2)).transpose(1, 2)
        x_dec_t2v = self.dec_t2v(x_dec)
        x_dec = torch.cat([x_dec, x_dec_t2v], dim=-1)
        x_dec = res_dec + self.dec_cnn(x_dec)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        ele_out = self.projection1(dec_out)[:, -self.pred_len:, :]
        cool_out = self.projection2(dec_out)[:, -self.pred_len:, :]
        heat_out = self.projection3(dec_out)[:, -self.pred_len:, :]

        outputs = torch.cat([ele_out, cool_out, heat_out], dim=-1)

        if self.output_attention:
            return outputs, attns
        else:
            return outputs
