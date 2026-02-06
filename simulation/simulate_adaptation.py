import torch
import pandas as pd
from models.transformer_silent_adaptation import SilentAdaptationTransformer

df = pd.read_csv("data/synthetic_silent_adaptation.csv")

features = [
    "treatment_intensity",
    "silent_adaptation",
    "tumor_burden"
]

def get_sample(df, pid, seq_len=10):
    sub = df[df.patient_id == pid][features].values
    return torch.tensor(sub[:seq_len]).unsqueeze(0)

model = SilentAdaptationTransformer(feature_dim=3)
sample = get_sample(df, pid=5)

risk, attn = model(sample)
print("Silent adaptation risk:", risk.item())
