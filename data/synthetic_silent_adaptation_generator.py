import numpy as np
import pandas as pd
import os

def generate_data(n_patients=300, timesteps=40):
    np.random.seed(0)
    rows = []

    for pid in range(n_patients):
        tumor_burden = np.random.uniform(0.2, 0.4)
        adaptation_signal = 0.0
        resistance = 0

        for t in range(timesteps):
            treatment = np.random.uniform(0.4, 1.0)

            # Sessiz adaptasyon yavaş ve görünmez
            adaptation_signal += np.random.uniform(0.0, 0.03)

            tumor_burden += (
                -treatment * 0.04
                + adaptation_signal * 0.02
                + np.random.normal(0, 0.01)
            )

            tumor_burden = np.clip(tumor_burden, 0, 1)

            # Direnç geç gelir
            if adaptation_signal > 0.8:
                resistance = 1

            rows.append([
                pid, t, treatment,
                adaptation_signal,
                tumor_burden,
                resistance
            ])

    df = pd.DataFrame(rows, columns=[
        "patient_id",
        "time",
        "treatment_intensity",
        "silent_adaptation",
        "tumor_burden",
        "resistance"
    ])

    output_path = os.path.join("data", "synthetic_silent_adaptation.csv")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    generate_data()
