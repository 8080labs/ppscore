# %% [markdown]
# ## Determining the best naive predictor for the f1 score
# - If there are 2 classes that are skewed, then the most common value is often slightly better than the random guess
# - If there are 4 classes that are skewed, then the random value is often slightly better than the most common value
# - If the classes (2 or 4) are balanced, then the random guess is usually significantly better than the most common value.
#
# Summing up, random values are usually preferred over the most common value.
#
# However, the best baseline is the maximum of the f1_score of the most common value and random values.

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# %%
df = pd.DataFrame(
    {
        "boolean_equal": np.random.choice(["yes", "no"], 1000),
        "boolean_skewed": np.random.choice(["yes", "yes", "yes", "no"], 1000),
        "multicat_equal": np.random.choice(["cat1", "cat2", "cat3", "cat4"], 1000),
        "multicat_skewed": np.random.choice(
            ["cat1", "cat1", "cat1", "cat1", "cat2", "cat2", "cat3", "cat4"], 1000
        ),
    }
)


# %%
def f1_score_most_common(series, value):
    return f1_score(series, np.random.choice([value], 1000), average="weighted")


# %%
def f1_score_random(series):
    return f1_score(series, series.sample(frac=1), average="weighted")


# %% [markdown]
# ### Boolean equal
# - Random is better than most common

# %%
f1_score_most_common(df["boolean_equal"], "yes")

# %%
f1_score_random(df["boolean_equal"])

# %% [markdown]
# ### Boolean skewed
# - Most common is usually better than random but they are in the same ball park

# %%
f1_score_most_common(df["boolean_skewed"], "yes")

# %%
f1_score_random(df["boolean_skewed"])

# %% [markdown]
# ### Multicat equal
# - Random is better than most common

# %%
f1_score_most_common(df["multicat_equal"], "cat1")

# %%
f1_score_random(df["multicat_equal"])

# %% [markdown]
# ### Multicat skewed
# - Random is usually better than most common but they are in the same ballpark

# %%
f1_score_most_common(df["multicat_skewed"], "cat1")

# %%
f1_score_random(df["multicat_skewed"])
