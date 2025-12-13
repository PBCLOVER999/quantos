# engine/factor_manager.py

import pandas as pd
from factors.library.basic_factors import generate_basic_factors


def run_factor_manager():
    print("[FactorManager] Computing factors...")

    df = generate_basic_factors()

    output_path = "data/features/basic_factors.csv"
    df.to_csv(output_path, index=False)

    print(f"[FactorManager] Saved factors to {output_path}")
    print(df.head())
