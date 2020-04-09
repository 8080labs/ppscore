# %%
import pandas as pd
import numpy as np
import seaborn as sns

import ppscore as pps


# %%
def heatmap(df):
    return sns.heatmap(df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)


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
