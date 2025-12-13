# QuantOS Data Ingest Service

import pandas as pd
import numpy as np

def run_data_ingest() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))

    return pd.DataFrame({"date": dates, "price": price})
