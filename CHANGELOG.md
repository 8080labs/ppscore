# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [dev] (unreleased)
- ...

## [1.2.0]
__Improvements:__
- Handle `pd.Series` that are `IntegerArray`s e.g. dtype `Int64`

## [1.1.1]
__Improvements:__
- Add installation of requirements

## [1.1.0]
__Default changes:__
- In `pps.score` changed the default of `random_seed` to `123` in order to enable reproducible results when comparing scores that are generated via `pps.score` or `pps.matrix`. The old behavior of drawing a random_seed per default was confusing. ([#32](https://github.com/8080labs/ppscore/issues/32))

__Improvements:__
- Improve error message when using the wrong API ([#31](https://github.com/8080labs/ppscore/issues/31))
- Automatic error catching for convenience with optional error inspection/debugging: added `catch_errors` to `pps.score` and added corresponding case `unknown_error`

## [1.0.0]
__Breaking changes:__
- The case of the calculation for `regression` or `classification` __only depends on the data types__ and not the cardinality of a column any more. Also, it is not possible any more to pass the `task` to `pps.score`. This removes confusion about the inference of the final task. Some other special cases like `feature_is_id` still review the cardinality of a column.
- The return format of `pps.matrix` changed to a tidy dataframe
- Changed the format of the PPS dict. It now includes information about errors and special cases in the `case` field. Also, there is `is_valid_score`

__Improvements:__
- Added new kwargs to `pps.score` e.g. `cross_validation`, `random_seed` and `invalid_score`
- Added error messages when the input arguments are invalid
- Added more tests
- Added CHANGELOG
