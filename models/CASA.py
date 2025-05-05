import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreNetwork(nn.Module):
    def __init__(self, enc_in, d_model, kernel=3):
        super(ScoreNetwork, self).__init__()

        assert kernel % 2 == 1, "Kernel size must be an odd number"
        pad = kernel // 2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=enc_in, out_channels=int(d_model / 4), kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model / 4), out_channels=int(d_model / 2), kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU(),
            nn.Conv1d(in_channels=int(d_model / 2), out_channels=d_model, kernel_size=kernel, stride=2, padding=pad),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=d_model, out_channels=int(d_model / 2), kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=int(d_model / 2), out_channels=int(d_model / 4), kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=int(d_model / 4), out_channels=enc_in, kernel_size=kernel, stride=2, padding=pad, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        skip = x
        decoded = self.decoder(encoded)
        output = decoded + skip[:, :, :decoded.size(2)]  # Adjusting the size if needed
        
        return output


class MLP_attention(nn.Module):
    def __init__(self, d_model, enc_in, kernel):
        super(MLP_attention, self).__init__()
        self.value = nn.Linear(d_model, d_model)
        self.score = ScoreNetwork(enc_in, d_model, kernel)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, L = x.shape
        x1 = torch.softmax(self.score(x), dim = 1) * self.value(x) + x
        x1 = self.norm(x1.reshape(-1, L)).reshape(B, N, L)
        # return self.mlp(x1) + x1
        return self.norm((self.mlp(x1) + x1).reshape(-1, L)).reshape(B, N, L)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.model = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            *[
                nn.Sequential(
                    MLP_attention(configs.d_model, configs.enc_in, configs.kernel),
                    nn.ReLU()
                )
                for _ in range(configs.e_layers)
            ],
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # instance norm
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # instance denorm
        y = y * torch.sqrt(seq_var) + seq_mean

        return y
