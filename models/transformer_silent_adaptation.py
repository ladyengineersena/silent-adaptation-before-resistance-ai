import torch
import torch.nn as nn

class SilentAdaptationTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, layers=3):
        super().__init__()

        self.embedding = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.attention = nn.Linear(d_model, 1)
        self.output = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        encoded = self.encoder(x)

        attn_weights = torch.softmax(
            self.attention(encoded), dim=1
        )

        context = (attn_weights * encoded).sum(dim=1)
        silent_risk = self.sigmoid(self.output(context))

        return silent_risk, attn_weights
