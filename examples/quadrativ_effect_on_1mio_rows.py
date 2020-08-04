# %%
import pandas as pd
import numpy as np
import seaborn as sns

import ppscore as pps


# %%
def heatmap(df):
    df = df[["x", "y", "ppscore"]].pivot(columns="x", index="y", values="ppscore")
    ax = sns.heatmap(df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    ax.set_title("PPS matrix")
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
    return ax


# %%
df = pd.DataFrame()
df["x"] = np.random.uniform(-2, 2, 1_000_000)
df["error"] = np.random.uniform(-0.5, 0.5, 1_000_000)
df["y"] = df["x"] * df["x"] + df["error"]

# %%
sns.scatterplot(x="x", y="y", data=df.sample(10_000))

# %%
matrix = pps.matrix(df)

# %%
matrix

# %%
heatmap(matrix)

# %%
pps.score(df, "x", "y")

# %%
