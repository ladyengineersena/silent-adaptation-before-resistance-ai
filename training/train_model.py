import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from models.transformer_silent_adaptation import SilentAdaptationTransformer

df = pd.read_csv("data/synthetic_silent_adaptation.csv")

features = [
    "treatment_intensity",
    "silent_adaptation",
    "tumor_burden"
]

def make_sequences(df, seq_len=10):
    X, y = [], []
    for pid in df.patient_id.unique():
        sub = df[df.patient_id == pid]
        vals = sub[features].values
        res = sub["resistance"].values
        for i in range(len(vals) - seq_len):
            X.append(vals[i:i+seq_len])
            y.append(res[i+seq_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = make_sequences(df)

model = SilentAdaptationTransformer(feature_dim=X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

for epoch in range(15):
    optimizer.zero_grad()
    preds, _ = model(X)
    loss = loss_fn(preds.squeeze(), y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} | Loss {loss.item():.4f}")
