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
    # each check is in the same order as in the original implementation
    from ppscore.calculation import _infer_task

    df = pd.read_csv("examples/titanic.csv")

    assert _infer_task(df, "Age", "Age") == "predict_itself"

    df["constant"] = 1
    assert _infer_task(df, "Age", "constant") == "predict_constant"

    assert _infer_task(df, "Age", "Survived") == "classification"

    df = df.reset_index()
    df["id"] = df["index"].astype(str)
    assert _infer_task(df, "Age", "id") == "predict_id"

    # classification because numeric but few categories
    assert _infer_task(df, "Age", "SibSp") == "classification"

    df["Pclass_category"] = df["Pclass"].astype("category")
    assert _infer_task(df, "Age", "Pclass_category") == "classification"

    df["Pclass_datetime"] = pd.to_datetime(df["Pclass"], infer_datetime_format=True)
    with pytest.raises(Exception):
        pps.score(df, "Age", "Pclass_datetime")

    assert _infer_task(df, "Survived", "Age") == "regression"


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
    with pytest.raises(Exception):
        pps.score(df, "nan", "y")

    assert pps.score(df, "x", "y", "regression")["task"] == "regression"

    assert pps.score(df, "x", "constant")["task"] == "predict_constant"
    assert pps.score(df, "x", "x")["task"] == "predict_itself"
    assert pps.score(df, "x", "id")["task"] == "predict_id"

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

    result_df = pps.predictors(df, y)
    assert isinstance(result_df, pd.DataFrame)
    assert not y in result_df.index

    list_of_dicts = pps.predictors(df, y, output="list")
    assert isinstance(list_of_dicts, list)
    assert isinstance(list_of_dicts[0], dict)


def test_matrix():
    df = pd.read_csv("examples/titanic.csv")
    df = df[["Age", "Survived"]]

    assert isinstance(pps.matrix(df), pd.DataFrame)
    assert isinstance(pps.matrix(df, output="dict"), dict)

    # matrix catches single score errors under the hood
    df["Age_datetime"] = pd.to_datetime(df["Age"], infer_datetime_format=True)
    assert pps.matrix(df[["Survived", "Age_datetime"]])["Survived"]["Age_datetime"] == 0
