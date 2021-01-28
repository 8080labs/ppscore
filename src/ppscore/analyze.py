import pandas as pd
from .calculation import matrix
from typing import Optional
import seaborn as sns
import matplotlib.pyplot as plt


def nonlinear_features(
        df: pd.DataFrame, pps_matrix: Optional[pd.DataFrame] = None,
        pos_border: float = 0.5, score_gap: float = 0.4, plot_nonlinear_features: bool = False,
        *args, **kwargs
) -> pd.DataFrame:
    """
    args and kwargs go to matrix calculations function

    :param plot_nonlinear_features:
    :param score_gap:
    :param pos_border:
    :param df: input DataFrame
    :param pps_matrix: optimize calculations if you already calculate it
    :return:
    """
    if pps_matrix is None:
        pps_matrix = matrix(df, *args, **kwargs)
    pps_matrix = pps_matrix[["x", "y", "ppscore"]]

    corr_matrix = df \
        .corr() \
        .abs() \
        .unstack() \
        .reset_index(name='correlation') \
        .rename(columns={
            "level_0": "x",
            "level_1": "y"
        }) \
        .merge(pps_matrix, on=['x', 'y'], how='left') \
        .dropna() \
        .assign(difference=lambda df: df['ppscore'] - df['correlation']) \
        .sort_values(by=['difference'], ascending=False)

    if plot_nonlinear_features:
        corr_matrix = corr_matrix[
            (corr_matrix['difference'].gt(score_gap) & corr_matrix['ppscore'].gt(pos_border)) |
            (corr_matrix['difference'].lt(-score_gap) & corr_matrix['ppscore'].lt(pos_border))
        ]

        for i, row in corr_matrix.reset_index(drop=True).iterrows():
            ax = sns.jointplot(
                data=df,
                x=row['x'],
                y=row['y'],
                kind='reg',
            )
            ax.fig.suptitle(f"correlation={row['correlation']:.3f}, ppscore={row['ppscore']:.3f}")
            ax.fig.tight_layout()

        plt.show()

    return corr_matrix
