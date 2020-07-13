# ppscore - a Python implementation of the Predictive Power Score (PPS)

### From the makers of [bamboolib](https://bamboolib.com)


__If you don't know yet what the Predictive Power Score is, please read the following blog post:__

__[RIP correlation. Introducing the Predictive Power Score](https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598)__

The PPS is an asymmetric, data-type-agnostic score that can detect linear or non-linear relationships between two columns. The score ranges from 0 (no predictive power) to 1 (perfect predictive power). It can be used as an alternative to the correlation (matrix).


- [Installation](#installation)
- [Getting started](#getting-started)
- [API](#api)
- [Calculation of the PPS](#calculation-of-the-pps)
- [About](#about)


## Installation

> You need Python 3.6 or above.

From the terminal (or Anaconda prompt in Windows), enter:

```bash
pip install ppscore
```


## Getting started

First, let's create some data:

```python
import pandas as pd
import numpy as np
import ppscore as pps

df = pd.DataFrame()
df["x"] = np.random.uniform(-2, 2, 1_000_000)
df["error"] = np.random.uniform(-0.5, 0.5, 1_000_000)
df["y"] = df["x"] * df["x"] + df["error"]
```

Based on the dataframe we can calculate the PPS of x predicting y:

```python
pps.score(df, "x", "y")
```

We can calculate the PPS of all the predictors in the dataframe against a target y

```python
pps.predictors(df, "y")
```

Here is how we can calculate the PPS matrix between all columns:

```python
pps.matrix(df)
```

For the visualization of the PPS matrix you might want to use seaborn or your favorite viz library:

```python
import seaborn as sns
df_matrix = pps.matrix(df)
sns.heatmap(df_matrix, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
```

Similarly, we can also plot the predictors

```python
import seaborn as sns
df_predictors = pps.predictors(df, y="y")
sns.barplot(data=df_predictors, x="x", y="ppscore")
```



## API

### ppscore.score(df, x, y, task=None, sample=5000)

Calculate the Predictive Power Score (PPS) for "x predicts y"

- The score always ranges from 0 to 1 and is data-type agnostic.

- A score of 0 means that the column x cannot predict the column y better than a naive baseline model.

- A score of 1 means that the column x can perfectly predict the column y given the model.

- A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the baseline model.


#### Parameters

- __df__ : pandas.DataFrame
    - Dataframe that contains the columns x and y
- __x__ : str
    - Name of the column x which acts as the feature
- __y__ : str
    - Name of the column y which acts as the target
- __task__ : str, default ``None``
    - Name of the prediction task, e.g. ``classification`` or ``regression``.
    If the task is not specified, it is infered based on the y column
    The task determines which model and evaluation score is used for the PPS
- __sample__ : int or ``None``
    - Number of rows for sampling. The sampling decreases the calculation time of the PPS.
    If ``None`` there will be no sampling.

#### Returns

- __Dict__:
    - A dict that contains multiple fields about the resulting PPS.
    The dict enables introspection into the calculations that have been performed under the hood


### ppscore.predictors(df, y, output="df", sorted=True, **kwargs)

Calculate the Predictive Power Score (PPS) for all columns in the dataframe against a target (y) column

#### Parameters
- __df__ : pandas.DataFrame
    - The dataframe that contains the data
- __y__ : str
    - Name of the column y which acts as the target
- __output__ : str - potential values: "df", "list"
    - Control the type of the output. Either return a df or a list with all the PPS score dicts
- __sorted__ : bool
    - Whether or not to sort the output dataframe/list
- __kwargs__ :
    - Other key-word arguments that shall be forwarded to the pps.score method

#### Returns

- __pandas.DataFrame__ or list of PPS dict:
    - Either returns a df or a list of all the PPS dicts. This can be influenced by the output argument


### ppscore.matrix(df, output="df", **kwargs)

Calculate the Predictive Power Score (PPS) matrix for all columns in the dataframe

#### Parameters

- __df__ : pandas.DataFrame
    - The dataframe that contains the data
- __output__ : str - potential values: "df", "dict"
    - Control the type of the output. Either return a df or a dict with all the PPS dicts arranged by the target column
- __kwargs__ :
    - Other key-word arguments that shall be forwarded to the pps.score method

#### Returns

- __pandas.DataFrame__ or __Dict__:
    - Either returns a df or a dict with all the PPS dicts arranged by the target column. This can be influenced by the output argument


## Calculation of the PPS

> If you are uncertain about some details, feel free to jump into the code to have a look at the exact implementation

There are multiple ways how you can calculate the PPS. The ppscore package provides a sample implementation that is based on the following calculations:

- The score is calculated using only 1 feature trying to predict the target column. This means there are no interaction effects between the scores of various features. Note that this is in contrast to feature importance
- The score is calculated on the test sets of a 4-fold crossvalidation (number is adjustable via `ppscore.CV_ITERATIONS`). For classification, stratifiedKFold is used. For regression, normal KFold. Please note that this sampling might not be valid for time series data sets
- All rows which have a missing value in the feature or the target column are dropped
- In case that the dataset has more than 5,000 rows the score is only calculated on a random subset of 5,000 rows with a fixed random seed (`ppscore.RANDOM_SEED`). You can adjust the number of rows or skip this sampling via the API. However, in most scenarios the results will be very similar
- There is no grid search for optimal model parameters


### Learning algorithm

As a learning algorithm, we currently use a Decision Tree because the Decision Tree has the following properties:
- can detect any non-linear bivariate relationship
- good predictive power in a wide variety of use cases
- low requirements for feature preprocessing
- robust model which can handle outliers and does not easily overfit
- can be used for classification and regression
- can be calculated quicker than many other algorithms

We differentiate the exact implementation based on the data type of the target column:
- If the target column is numeric, we use the sklearn.DecisionTreeRegressor
- If the target column is categoric, we use the sklearn.DecisionTreeClassifier

> Please note that we prefer a general good performance on a wide variety of use cases over better performance in some narrow use cases. If you have a proposal for a better/different learning algorithm, please open an issue

However, please note why we actively decided against the following algorithms:

- Correlation or Linear Regression: cannot detect non-linear bivariate relationships without extensive preprocessing
- GAMs: might have problems with very unsmooth functions
- SVM: potentially bad performance if the wrong kernel is selected
- Random Forest/Gradient Boosted Tree: slower than a single Decision Tree
- Neural Networks and Deep Learning: slower calculation than a Decision Tree and also needs more feature preprocessing

### Data preprocessing

Even though the Decision Tree is a very flexible learning algorithm, we need to perform the following preprocessing steps if a column represents categoric values - that means it has the pandas dtype `object`, `category`, `string` or `boolean`.‌
- If the target column is categoric, we use the sklearn.LabelEncoder​
- If the feature column is categoric, we use the sklearn.OneHotEncoder​


### Inference of the prediction task

The choice of the task (classification or regression) has an influence on the final PPS and thus it is important how the task is chosen. If you calculate a single score, you can specify the task via the API. If you do not specify the task, the task is inferred as follows.

A __classification__ is inferred if one of the following conditions meet:
- the target has the dtype `object`, `category`, `string` or `boolean`
- the target only has two unique values
- the target is numeric but has less than 15 unique values. This breakpoint can be overridden via the constant `ppscore.NUMERIC_AS_CATEGORIC_BREAKPOINT`

Otherwise, the task is inferred as __regression__ if the dtype is numeric (float or integer).


### Tasks and their score metrics​

Based on the data type and cardinality of the target column, ppscore assumes either the task of a classification or regression. Each task uses a different evaluation score for calculating the final predictive power score (PPS).

#### Regression

In case of an regression, the ppscore uses the mean absolute error (MAE) as the underlying evaluation metric (MAE_model). The best possible score of the MAE is 0 and higher is worse. As a baseline score, we calculate the MAE of a naive model (MAE_naive) that always predicts the median of the target column. The PPS is the result of the following normalization (and never smaller than 0):
> PPS = 1 - (MAE_model / MAE_naive)

#### Classification

If the task is a classification, we compute the weighted F1 score (wF1) as the underlying evaluation metric (F1_model). The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The weighted F1 takes into account the precision and recall of all classes weighted by their support as described [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). As a baseline score (F1_naive), we calculate the weighted F1 score for a model that always predicts the most common class of the target column (F1_most_common) and a model that predicts random values (F1_random). F1_naive is set to the maximum of F1_most_common and F1_random. The PPS is the result of the following normalization (and never smaller than 0):
> PPS = (F1_model - F1_naive) / (1 - F1_naive)


## About
ppscore is developed by [8080 Labs](https://8080labs.com) - we create tools for Python Data Scientists. If you like `ppscore`, please check out our other project [bamboolib](https://bamboolib.com)
