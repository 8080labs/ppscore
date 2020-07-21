# # -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np

import ppscore as pps


def test__normalized_f1_score():
    from ppscore.calculation import _normalized_f1_score

    assert _normalized_f1_score(0.4, 0.5) == 0
    assert _normalized_f1_score(0.75, 0.5) == 0.5


def test__normalized_mae_score():
    from ppscore.calculation import _normalized_mae_score

    assert _normalized_mae_score(10, 5) == 0
    assert _normalized_mae_score(5, 10) == 0.5


def test__infer_task():
    from ppscore.calculation import _infer_task

    df = pd.read_csv("examples/titanic.csv")
    df = df.rename(
        columns={
            "Age": "Age_float",
            "Pclass": "Pclass_integer",
            "Survived": "Survived_integer",
            "Ticket": "Ticket_object",
            "Name": "Name_object_id",
        }
    )

    df["x"] = 1  # x is irrelevant for this test
    df["constant"] = 1
    df["Pclass_category"] = df["Pclass_integer"].astype("category")
    df["Pclass_datetime"] = pd.to_datetime(
        df["Pclass_integer"], infer_datetime_format=True
    )
    df["Survived_boolean"] = df["Survived_integer"].astype(bool)
    df["Cabin_string"] = pd.Series(df["Cabin"].apply(str), dtype="string")

    # check special types
    assert _infer_task(df, "x", "x") == "predict_itself"
    assert _infer_task(df, "x", "constant") == "predict_constant"
    assert _infer_task(df, "x", "Name_object_id") == "predict_id"

    # check regression
    assert _infer_task(df, "x", "Age_float") == "regression"
    assert _infer_task(df, "x", "Pclass_integer") == "regression"

    # check classification
    assert _infer_task(df, "x", "Pclass_category") == "classification"
    assert _infer_task(df, "x", "Survived_boolean") == "classification"
    assert _infer_task(df, "x", "Ticket_object") == "classification"
    assert _infer_task(df, "x", "Cabin_string") == "classification"

    # datetime columns are not supported
    with pytest.raises(TypeError):
        pps.score(df, "x", "Pclass_datetime")


def test__maybe_sample():
    from ppscore.calculation import _maybe_sample

    df = pd.read_csv("examples/titanic.csv")
    assert len(_maybe_sample(df, 10)) == 10


def test_score():
    df = pd.DataFrame()
    df["x"] = np.random.uniform(-2, 2, 1_000)
    df["error"] = np.random.uniform(-0.5, 0.5, 1_000)
    df["y"] = df["x"] * df["x"] + df["error"]

    df["constant"] = 1
    df = df.reset_index()
    df["id"] = df["index"].astype(str)

    df["x_greater_0_boolean"] = df["x"] > 0
    # df["x_greater_0_string"] = df["x_greater_0_boolean"].astype(str)
    df["x_greater_0_string"] = pd.Series(
        df["x_greater_0_boolean"].apply(str), dtype="string"
    )
    df["x_greater_0_string_object"] = df["x_greater_0_string"].astype("object")
    df["x_greater_0_string_category"] = df["x_greater_0_string"].astype("category")

    df["x_greater_0_boolean_object"] = df["x_greater_0_boolean"].astype("object")
    df["x_greater_0_boolean_category"] = df["x_greater_0_boolean"].astype("category")

    df["nan"] = np.nan

    # check input types
    with pytest.raises(TypeError):
        numpy_array = np.random.randn(10, 10)  # not a DataFrame
        pps.score(numpy_array, "x", "y")

    with pytest.raises(ValueError):
        pps.score(df, "x_column_that_does_not_exist", "y")

    with pytest.raises(ValueError):
        pps.score(df, "x", "y_column_that_does_not_exist")

    with pytest.raises(Exception):
        # After dropping missing values, there are no valid rows left
        pps.score(df, "nan", "y")

    with pytest.raises(AttributeError):
        # the task argument is not supported any more
        pps.score(df, "x", "y", task="classification")

    # check tasks
    assert pps.score(df, "x", "y")["task"] == "regression"
    assert pps.score(df, "x", "x_greater_0_string")["task"] == "classification"
    assert pps.score(df, "x", "constant")["task"] == "predict_constant"
    assert pps.score(df, "x", "x")["task"] == "predict_itself"
    assert pps.score(df, "x", "id")["task"] == "predict_id"

    # check scores
    # feature is id
    assert pps.score(df, "id", "y")["ppscore"] == 0

    # numeric feature and target
    assert pps.score(df, "x", "y")["ppscore"] > 0.5
    assert pps.score(df, "y", "x")["ppscore"] < 0.05

    # boolean feature or target
    assert pps.score(df, "x", "x_greater_0_boolean")["ppscore"] > 0.6
    assert pps.score(df, "x_greater_0_boolean", "x")["ppscore"] < 0.6

    # string feature or target
    assert pps.score(df, "x", "x_greater_0_string")["ppscore"] > 0.6
    assert pps.score(df, "x_greater_0_string", "x")["ppscore"] < 0.6

    # object feature or target
    assert pps.score(df, "x", "x_greater_0_string_object")["ppscore"] > 0.6
    assert pps.score(df, "x_greater_0_string_object", "x")["ppscore"] < 0.6

    # category feature or target
    assert pps.score(df, "x", "x_greater_0_string_category")["ppscore"] > 0.6
    assert pps.score(df, "x_greater_0_string_category", "x")["ppscore"] < 0.6

    # object feature or target
    assert pps.score(df, "x", "x_greater_0_boolean_object")["ppscore"] > 0.6
    assert pps.score(df, "x_greater_0_boolean_object", "x")["ppscore"] < 0.6

    # category feature or target
    assert pps.score(df, "x", "x_greater_0_boolean_category")["ppscore"] > 0.6
    assert pps.score(df, "x_greater_0_boolean_category", "x")["ppscore"] < 0.6


def test_predictors():
    y = "Survived"
    df = pd.read_csv("examples/titanic.csv")
    df = df[["Age", y]]

    # check input types
    with pytest.raises(TypeError):
        numpy_array = np.random.randn(10, 10)  # not a DataFrame
        pps.predictors(numpy_array, y)

    with pytest.raises(ValueError):
        pps.predictors(df, "y_column_that_does_not_exist")

    with pytest.raises(ValueError):
        pps.predictors(df, y, output="invalid_output_type")

    with pytest.raises(ValueError):
        pps.predictors(df, y, sorted="invalid_value_for_sorted")

    # check return types
    result_df = pps.predictors(df, y)
    assert isinstance(result_df, pd.DataFrame)
    assert not y in result_df.index

    list_of_dicts = pps.predictors(df, y, output="list")
    assert isinstance(list_of_dicts, list)
    assert isinstance(list_of_dicts[0], dict)

    # the underlying calculations are tested as part of test_score


def test_matrix():
    df = pd.read_csv("examples/titanic.csv")
    df = df[["Age", "Survived"]]
    df["Age_datetime"] = pd.to_datetime(df["Age"], infer_datetime_format=True)

    # check input types
    with pytest.raises(TypeError):
        numpy_array = np.random.randn(10, 10)  # not a DataFrame
        pps.matrix(numpy_array)

    with pytest.raises(ValueError):
        pps.matrix(df, output="invalid_output_type")

    # check return types
    assert isinstance(pps.matrix(df), pd.DataFrame)
    assert isinstance(pps.matrix(df, output="dict"), dict)

    # matrix catches single score errors under the hood
    assert pps.matrix(df[["Survived", "Age_datetime"]])["Survived"]["Age_datetime"] == 0
