import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def calculate_center_of_mass(df):
    """
    Compute per-sequence, per-cell-type center of mass from sort-seq bin counts.

    Expects columns: seq, cell_type, replicate, "1", "2", "3", "4".
    Returns a DataFrame indexed by seq with cell types as columns.
    """
    grouped = df.groupby(["seq", "cell_type", "replicate"]).sum().reset_index()
    bins = np.arange(1, 5)
    cpm = grouped[["1", "2", "3", "4"]]
    grouped["center_of_mass"] = (cpm * bins).sum(axis=1) / cpm.sum(axis=1)
    final = grouped.groupby(["seq", "cell_type"])["center_of_mass"].mean().reset_index()
    return final.pivot(index="seq", columns="cell_type", values="center_of_mass")


def calculate_correlations_of_patterns(df1_, df2_, method="spearman"):
    """
    For each sequence present in both df1_ and df2_, compute the correlation
    between cell-type expression vectors.

    Args:
        df1_: DataFrame indexed by sequence, columns are cell types.
        df2_: DataFrame with a 'seq' column and cell-type columns.
        method: "spearman" or "pearson".

    Returns:
        pd.DataFrame with columns [seq, correlation].
    """
    df1 = df1_.copy(deep=True)
    df2 = df2_.copy(deep=True)

    df1_num_cols = df1.select_dtypes(include=["float64", "int64"]).columns
    df2_num_cols = df2.select_dtypes(include=["float64", "int64"]).columns
    common_cols = list(set(df1_num_cols).intersection(set(df2_num_cols)))

    if not common_cols:
        raise ValueError("No common numerical columns found between the dataframes.")

    df1.index = df1.index.str.upper()
    df2["seq"] = df2["seq"].str.upper()
    merged = pd.merge(df1, df2, left_index=True, right_on="seq", how="inner", suffixes=("_df1", "_df2"))

    def calc_corr(row):
        v1 = np.array(row[[c + "_df1" for c in common_cols]])
        v2 = np.array(row[[c + "_df2" for c in common_cols]])
        if len(v1) < 2:
            return np.nan
        if method == "spearman":
            return spearmanr(v1, v2).statistic
        return np.corrcoef(v1, v2)[0, 1]

    correlations = merged.apply(
        lambda row: {"seq": row["seq"], "correlation": calc_corr(row)}, axis=1
    ).tolist()
    return pd.DataFrame(correlations).dropna()


def plot_expression_radar(
    df,
    title="",
    utr="5",
    cols_to_draw=None,
    cols_to_drop=None,
):
    """
    Radar (spider) plots for each row in df, showing expression across cell types.

    Args:
        df: DataFrame with cell-type columns and metadata.
        title: Overall figure title.
        utr: "5" or "3" - controls subtitle format.
        cols_to_draw: Explicit list of columns to plot. If None, inferred by dropping cols_to_drop.
        cols_to_drop: Columns to exclude when cols_to_draw is None.
    """
    if cols_to_drop is None:
        cols_to_drop = [
            "seq", "tau", "Cluster", "Unified_Cluster",
            "subs", "max_diff_x", "max_diff_pair_pred",
            "source", "max_diff_y", "max_diff_pair_exp",
        ]

    n_rows = len(df)
    fig, axes = plt.subplots(1, n_rows, figsize=(4 * n_rows, 4), subplot_kw=dict(projection="polar"))
    fig.suptitle(title)

    if n_rows == 1:
        axes = [axes]

    cols_to_plot = cols_to_draw if cols_to_draw else [c for c in df.columns if c not in cols_to_drop]
    angles = np.linspace(0, 2 * np.pi, len(cols_to_plot), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = np.concatenate(([row[c] for c in cols_to_plot], [row[cols_to_plot[0]]]))
        axes[idx].plot(angles, values)
        axes[idx].fill(angles, values, alpha=0.25)
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(cols_to_plot)
        axes[idx].set_ylim(1, 4)
        if utr == "5":
            axes[idx].set_title(
                f"r={row['correlation']:.2f}, tau={row['tau']:.2f}, "
                f"min_dist={row['min_dist']}, {row['source']}\n{row['seq']}",
                fontsize=6,
            )
        else:
            axes[idx].set_title(
                f"r={row['correlation']:.2f}, tau={row['tau']:.2f}, "
                f"min_dist={row['min_dist']}, {row['source']}",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()


def plot_expression_barplot(df, cols_to_draw):
    """
    Bar plots for each row in df showing expression levels across cell types.

    Args:
        df: DataFrame with cell-type columns and optional Unified_Cluster column.
        cols_to_draw: List of cell-type column names to plot.
    """
    n_rows = len(df)
    fig, axes = plt.subplots(1, n_rows, figsize=(4 * n_rows, 4))
    if n_rows == 1:
        axes = [axes]

    x = np.arange(len(cols_to_draw))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[col] for col in cols_to_draw]
        axes[idx].bar(x, values)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(cols_to_draw, rotation=45)
        axes[idx].set_ylim(1.2, 3.8)
        if "Unified_Cluster" in row:
            axes[idx].set_title(f"Sequence {idx + 1}\nCluster: {row['Unified_Cluster']}")
        else:
            axes[idx].set_title(f"Sequence {idx + 1}")

    plt.tight_layout()
    plt.show()
