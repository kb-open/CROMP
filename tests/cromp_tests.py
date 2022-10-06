# CROMP testing pipeline
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

import gc, argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import lsq_linear
from cromp import CROMPTrain, CROMPPredict

def _mape(predictions, actuals):
    return abs((predictions - actuals) / actuals).mean()

def _test_1(df_train, df_test, target_col, feature_cols_in_asc_order,
                 min_gap_pct, lb, ub):
    model = CROMPTrain()
    ret_success = model.configure(df_train, target_col, feature_cols_in_asc_order, min_gap_pct, lb, ub)
    if ret_success:
        ret_success, wages = model.train()

    if ret_success:
        print("\nPredicted intercept from CROMP:", wages[0])
        print("Predicted coefficients from CROMP:", wages[1:])

        model = CROMPPredict(wages)
        ret_success, result = model.predict(df_test, feature_cols_in_asc_order)
        if ret_success:
            print("Predicted result from CROMP:\n", result)
            print("MAPE from CROMP:", _mape(result, df_test[target_col]))

def _test_2(df_train, df_test, target_col, feature_cols_in_asc_order):
    model = LinearRegression()
    model.fit(df_train[feature_cols_in_asc_order], df_train[target_col])
    print("\nPredicted intercept from MLR:", model.intercept_)
    print("Predicted coefficients from MLR:", model.coef_)

    result = model.predict(df_test[feature_cols_in_asc_order])
    print("Predicted result from MLR:\n", result)
    print("MAPE from MLR:", _mape(result, df_test[target_col]))

def _test_3(df_train, df_test, target_col, feature_cols_in_asc_order):
    model = LinearRegression(fit_intercept=False)
    model.fit(df_train[feature_cols_in_asc_order], df_train[target_col])
    print("\nPredicted coefficients from MLR without intercept:", model.coef_)

    result = model.predict(df_test[feature_cols_in_asc_order])
    print("Predicted result from MLR without intercept:\n", result)
    print("MAPE from MLR without intercept:", _mape(result, df_test[target_col]))

def _test_4(df_train, df_test, target_col, feature_cols_in_asc_order, lb, ub):
    model = lsq_linear(df_train[feature_cols_in_asc_order], df_train[target_col], bounds=(lb, ub))
    print("\nPredicted coefficients from Linear LSQ:", model.x)

    result = df_test.apply(lambda row: sum([x * y for x, y in zip(model.x, row[feature_cols_in_asc_order])]),
                           axis=1)
    print("Predicted result from Linear LSQ:\n", result)
    print("MAPE from Linear LSQ:", _mape(result, df_test[target_col]))

def _perform_benchmarking(num_training_samples):
    df = pd.read_excel("tests/data/benchmark_wage_data.xlsx", sheet_name="hc_data")
    
    if num_training_samples >= len(df):
        print("\nERROR! Number of training samples must be lower than the total number of rows in the entire dataset.\n")
        return
        
    df_train = df.iloc[:num_training_samples, :]
    df_test = df.iloc[num_training_samples:, :]

    target_col = 'TotalWageCost_Values'
    feature_cols_in_asc_order = ['HeadCount_DS_1', 'HeadCount_DS_3', 'HeadCount_Sr_DS_2', 'HeadCount_Sr_DS_1', 'HeadCount_DS_2']

    min_gap_pct = [0.13, 0.51, 0.09, 0.03]
    lb = [56, 64, 108, 97, 111]
    ub = [95, 106, 171, 160, 176]

    _test_1(df_train, df_test, target_col, feature_cols_in_asc_order, min_gap_pct, lb, ub)
    _test_2(df_train, df_test, target_col, feature_cols_in_asc_order)
    _test_3(df_train, df_test, target_col, feature_cols_in_asc_order)
    _test_4(df_train, df_test, target_col, feature_cols_in_asc_order, lb, ub)

def _perform_ut():
    df = pd.read_csv("tests/data/ames_house_prices_data.csv")
    df_train = df.iloc[:-50, :]
    df_test = df.iloc[-50:, :]

    target_col = 'SalePrice'
    feature_cols_in_asc_order = ['1stFlrSF', 'TotalBsmtSF', 'GrLivArea']

    min_gap_pct = 0.5
    lb = 0.0
    ub = 100.0

    _test_1(df_train, df_test, target_col, feature_cols_in_asc_order, min_gap_pct, lb, ub)
    _test_2(df_train, df_test, target_col, feature_cols_in_asc_order)
    _test_3(df_train, df_test, target_col, feature_cols_in_asc_order)
    _test_4(df_train, df_test, target_col, feature_cols_in_asc_order, lb, ub)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CROMP testing pipeline. Use -b option to select benchmarking test with n number of training samples.")
    
    parser.add_argument("-b", "--Benchmark", default=False, action="store_true", help="select benchmarking test")
    parser.add_argument("NumTrainSamples", metavar='n', type=int, nargs='?', help="specify number of training samples for benchmarking test")
    
    args = parser.parse_args()
    print("****", args)
    
    if args.Benchmark:
        if args.NumTrainSamples:
            print("\nBenchmarking test with {} training samples:".format(args.NumTrainSamples))
            print("\n===========================================\n")
            _perform_benchmarking(num_training_samples=args.NumTrainSamples)
        else:
            print("\nERROR! Benchmarking test requires number of training samples to be specified.\n")
    else:
        print("\nUnit test:")
        print("\n==========\n")
        _perform_ut()

    gc.collect()
    