# %% [markdown]
# ## Applying the PPS to the Titanic dataset
# - This script shows you how to apply the PPS to the Titanic dataset
# - If you want to execute the script yourself, you need to have valid installations of the packages ppscore, seaborn and pandas.

# %%
import pandas as pd
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
def corr_heatmap(df):
    ax = sns.heatmap(df, vmin=-1, vmax=1, cmap="BrBG", linewidths=0.5, annot=True)
    ax.set_title("Correlation matrix")
    return ax


# %%
df = pd.read_csv("titanic.csv")

# %% [markdown]
# ## Preparation of the Titanic dataset
# - Selecting a subset of columns
# - Changing some data types
# - Renaming the column names to be more clear

# %%
df = df[["Survived", "Pclass", "Sex", "Age", "Ticket", "Fare", "Embarked"]]
df = df.rename(columns={"Pclass": "Class"})
df = df.rename(columns={"Ticket": "TicketID"})
df = df.rename(columns={"Fare": "TicketPrice"})
df = df.rename(columns={"Embarked": "Port"})

# %% [markdown]
# ## Single Predictive Power Score
# - Answering the question: how well can Sex predict the Survival probability?

# %%
pps.score(df, "Sex", "Survived")

# %% [markdown]
# ## PPS matrix
# - Answering the question: which predictive patterns exist between the columns?

# %%
matrix = pps.matrix(df)

# %%
matrix

# %%
heatmap(matrix)

# %% [markdown]
# ## Correlation matrix
# - As a comparison to the PPS matrix

# %%
corr_heatmap(df.corr())

# %%
