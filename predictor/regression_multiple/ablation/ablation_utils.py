import os

import numpy as np
import pandas as pd
import scipy.stats as ss


def load_data(utr_type, prefix="."):
    if utr_type.lower() == "utr3":
        PATH_FROM = os.path.join(prefix, "UTR3_zscores_replicateagg.csv")
    elif utr_type.lower() == "utr5":
        PATH_FROM = os.path.join(prefix, "UTR5_zscores_replicateagg.csv")
    df = pd.read_csv(PATH_FROM)

    splits = dict(tuple(df.groupby('fold')))
    for split_df in splits.values():
        split_df.reset_index(drop=True, inplace=True)
    return splits


