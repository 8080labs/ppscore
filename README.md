# ppscore - a Python implementation of the Predictive Power Score (PPS)

### From the makers of [bamboolib - a GUI for pandas DataFrames](https://bamboolib.com)


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
pip install -U ppscore
```


## Getting started

> The examples refer to the newest version (1.2.0) of ppscore. [See changes](https://github.com/8080labs/ppscore/blob/master/CHANGELOG.md)

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

We can calculate the PPS of all the predictors in the dataframe against a target y:

```python
pps.predictors(df, "y")
```

Here is how we can calculate the PPS matrix between all columns:

```python
pps.matrix(df)
```


### Visualization of the results
For the visualization of the results you can use seaborn or your favorite viz library.

__Plotting the PPS predictors:__

```python
import seaborn as sns
predictors_df = pps.predictors(df, y="y")
sns.barplot(data=predictors_df, x="x", y="ppscore")
```

__Plotting the PPS matrix:__

(This needs some minor preprocessing because seaborn.heatmap unfortunately does not accept tidy data)

```python
import seaborn as sns
matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
```


## API

### ppscore.score(df, x, y, sample=5_000, cross_validation=4, random_seed=123, invalid_score=0, catch_errors=True)

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
- __sample__ : int or `None`
    - Number of rows for sampling. The sampling decreases the calculation time of the PPS.
    If `None` there will be no sampling.
- __cross_validation__ : int
    - Number of iterations during cross-validation. This has the following implications:
    For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same observation. If the limit is increased, the required minimum observations also increase. This is important, because this is the limit when sklearn will throw an error and the PPS cannot be calculated
- __random_seed__ : int or `None`
    - Random seed for the parts of the calculation that require random numbers, e.g. shuffling or sampling.
    If the value is set, the results will be reproducible. If the value is `None` a new random number is drawn at the start of each calculation.
- __invalid_score__ : any
    - The score that is returned when a calculation is not valid, e.g. because the data type was not supported.
- __catch_errors__ : bool
    - If `True` all errors will be catched and reported as `unknown_error` which ensures convenience. If `False` errors will be raised. This is helpful for inspecting and debugging errors.


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
    - Whether or not to sort the output dataframe/list by the ppscore
- __kwargs__ :
    - Other key-word arguments that shall be forwarded to the pps.score method, e.g. __sample__, __cross_validation__, __random_seed__, __invalid_score__, __catch_errors__

#### Returns

- __pandas.DataFrame__ or list of PPS dicts:
    - Either returns a df or a list of all the PPS dicts. This can be influenced by the output argument


### ppscore.matrix(df, output="df", sorted=False, **kwargs)

Calculate the Predictive Power Score (PPS) matrix for all columns in the dataframe

#### Parameters

- __df__ : pandas.DataFrame
    - The dataframe that contains the data
- __output__ : str - potential values: "df", "list"
    - Control the type of the output. Either return a df or a list with all the PPS score dicts
- __sorted__ : bool
    - Whether or not to sort the output dataframe/list by the ppscore
- __kwargs__ :
    - Other key-word arguments that shall be forwarded to the pps.score method, e.g. __sample__, __cross_validation__, __random_seed__, __invalid_score__, __catch_errors__

#### Returns

- __pandas.DataFrame__ or list of PPS dicts:
    - Either returns a df or a list of all the PPS dicts. This can be influenced by the output argument


## Calculation of the PPS

> If you are uncertain about some details, feel free to jump into the code to have a look at the exact implementation

There are multiple ways how you can calculate the PPS. The ppscore package provides a sample implementation that is based on the following calculations:

- The score is calculated using only 1 feature trying to predict the target column. This means there are no interaction effects between the scores of various features. Note that this is in contrast to feature importance
- The score is calculated on the test sets of a 4-fold cross-validation (number is adjustable via `cross_validation`). For classification, stratifiedKFold is used. For regression, normal KFold. Please note that __this sampling might not be valid for time series data sets__
- All rows which have a missing value in the feature or the target column are dropped
- In case that the dataset has more than 5,000 rows the score is only calculated on a random subset of 5,000 rows. You can adjust the number of rows or skip this sampling via `sample`. However, in most scenarios the results will be very similar
- There is no grid search for optimal model parameters
- The result might change between calculations because the calculation contains random elements, e.g. the sampling of the rows or the shuffling of the rows before cross-validation. If you want to make sure that your results are reproducible you can set the random seed (`random_seed`).
- If the score cannot be calculated, the package will not raise an error but return an object where `is_valid_score` is `False`. The reported score will be `invalid_score`. We chose this behavior because we want to give you a quick overview where significant predictive power exists without you having to handle errors or edge cases. However, when you want to explicitly handle the errors, you can still do so.

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
- If the target column is categoric, we use the `sklearn.LabelEncoder​`
- If the feature column is categoric, we use the `sklearn.OneHotEncoder​`


### Choosing the prediction case

> This logic was updated in version 1.0.0.

The choice of the case (`classification` or `regression`) has an influence on the final PPS and thus it is important that the correct case is chosen. The case is chosen based on the data types of the columns. That means, e.g. if you want to change the case from `regression` to `classification` that you have to change the data type from `float` to `string`.

Here are the two main cases:
- A __classification__ is chosen if the target has the dtype `object`, `category`, `string` or `boolean`
- A __regression__ is chosen if the target has the dtype `float` or `int`


### Cases and their score metrics​

Each case uses a different evaluation score for calculating the final predictive power score (PPS).

#### Regression

In case of an regression, the ppscore uses the mean absolute error (MAE) as the underlying evaluation metric (MAE_model). The best possible score of the MAE is 0 and higher is worse. As a baseline score, we calculate the MAE of a naive model (MAE_naive) that always predicts the median of the target column. The PPS is the result of the following normalization (and never smaller than 0):
> PPS = 1 - (MAE_model / MAE_naive)

#### Classification

If the task is a classification, we compute the weighted F1 score (wF1) as the underlying evaluation metric (F1_model). The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The weighted F1 takes into account the precision and recall of all classes weighted by their support as described [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). As a baseline score (F1_naive), we calculate the weighted F1 score for a model that always predicts the most common class of the target column (F1_most_common) and a model that predicts random values (F1_random). F1_naive is set to the maximum of F1_most_common and F1_random. The PPS is the result of the following normalization (and never smaller than 0):
> PPS = (F1_model - F1_naive) / (1 - F1_naive)

### Special cases

There are various cases in which the PPS can be defined without fitting a model to save computation time or in which the PPS cannot be calculated at all. Those cases are described below.

#### Valid scores
In the following cases, the PPS is defined but we can save ourselves the computation time:
- __feature_is_id__ means that the feature column is categoric (see above for __classification__) and that all categories appear only once. Such a feature can never predict a target during cross-validation and thus the PPS is 0.
- __target_is_id__ means that the target column is categoric (see above for __classification__) and that all categories appear only once. Thus, the PPS is 0 because an ID column cannot be predicted by any other column as part of a cross-validation. There still might be a 1 to 1 relationship but this is not detectable by the current implementation of the PPS.
- __target_is_constant__ means that the target column only has a single value and thus the PPS is 0 because any column and baseline can perfectly predict a column that only has a single value. Therefore, the feature does not add any predictive power and we want to communicate that.
- __predict_itself__ means that the feature and target columns are the same and thus the PPS is 1 because a column can always perfectly predict its own value. Also, this leads to the typical diagonal of 1 that we are used to from the correlation matrix.

#### Invalid scores and other errors
In the following cases, the PPS is not defined and the score is set to `invalid_score`:
- __target_is_datetime__ means that the target column has a datetime data type which is not supported. A possible solution might be to convert the target column to a string column.
- __target_data_type_not_supported__ means that the target column has a data type which is not supported. A possible solution might be to convert the target column to another data type.
- __empty_dataframe_after_dropping_na__ occurs when there are no valid rows left after rows with missing values have been dropped. A possible solution might be to replace the missing values with valid values.
- Last but not least, __unknown_error__ occurs for all other errors that might raise an exception. This case is only reported when `catch_errors` is `True`. If you want to inspect or debug the underlying error, please set `catch_errors` to `False`.


## About
ppscore is developed by [8080 Labs](https://8080labs.com) - we create tools for Python Data Scientists. If you like `ppscore` you might want to check out our other project [bamboolib - a GUI for pandas DataFrames](https://bamboolib.com)
