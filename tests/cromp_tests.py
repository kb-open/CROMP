# CROMP testing pipeline
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

import gc, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import lsq_linear
from cromp import CROMPTrain, CROMPPredict

def _mape(predictions, actuals):
    return abs((predictions - actuals) / actuals).mean()

def _test_1(df_train, df_test, target_col,
            feats_in_asc_order, min_gap_pct,
            feats_in_no_order,
            lb, ub):
    model = CROMPTrain()
    ret_success = model.config_constraints(feats_in_asc_order, min_gap_pct=min_gap_pct,\
                                           feats_in_no_order=feats_in_no_order,\
                                           lb=lb, ub=ub)
    if ret_success:
        ret_success, cromp_model = model.train(df_train, target_col)

    if ret_success:
        print("\nPredicted intercept from CROMP:", cromp_model['coeffs'][0])
        print("Predicted coefficients from CROMP:", cromp_model['coeffs'][1:])

        model = CROMPPredict(cromp_model)
        result = model.predict(df_test)
        if ret_success:
            print("Predicted result from CROMP:\n", result)
            print("MAPE from CROMP:", _mape(result, df_test[target_col]))

def _test_2(df_train, df_test, target_col, feats_in_asc_order):
    model = LinearRegression()
    model.fit(df_train[feats_in_asc_order], df_train[target_col])
    print("\nPredicted intercept from MLR:", model.intercept_)
    print("Predicted coefficients from MLR:", model.coef_)

    result = model.predict(df_test[feats_in_asc_order])
    print("Predicted result from MLR:\n", result)
    print("MAPE from MLR:", _mape(result, df_test[target_col]))

def _test_3(df_train, df_test, target_col, feats_in_asc_order):
    model = LinearRegression(fit_intercept=False)
    model.fit(df_train[feats_in_asc_order], df_train[target_col])
    print("\nPredicted coefficients from MLR without intercept:", model.coef_)

    result = model.predict(df_test[feats_in_asc_order])
    print("Predicted result from MLR without intercept:\n", result)
    print("MAPE from MLR without intercept:", _mape(result, df_test[target_col]))

def _test_4(df_train, df_test, target_col, feats_in_asc_order, lb, ub):
    model = lsq_linear(df_train[feats_in_asc_order], df_train[target_col], bounds=(lb, ub))
    print("\nPredicted coefficients from Linear LSQ:", model.x)

    result = df_test.apply(lambda row: sum([x * y for x, y in zip(model.x, row[feats_in_asc_order])]),
                           axis=1)
    print("Predicted result from Linear LSQ:\n", result)
    print("MAPE from Linear LSQ:", _mape(result, df_test[target_col]))

def _perform_benchmarking(num_training_samples, data_path:str=None):
    if not data_path:
        data_path = "tests/data/benchmark_wage_data.xlsx"
    df = pd.read_excel(data_path, sheet_name="hc_data")
    
    if num_training_samples >= len(df):
        print("\nERROR! Number of training samples must be lower than the total number of rows in the entire dataset.\n")
        return
        
    df_train = df.iloc[:num_training_samples, :]
    df_test = df.iloc[num_training_samples:, :]

    target_col = 'TotalWageCost_Values'
    feats_in_asc_order = ['HeadCount_DS_1', 'HeadCount_DS_3', 'HeadCount_Sr_DS_2', 'HeadCount_Sr_DS_1', 'HeadCount_DS_2']
    feats_in_no_order = []

    min_gap_pct = [0.13, 0.51, 0.09, 0.03]
    lb = [56, 64, 108, 97, 111]
    ub = [95, 106, 171, 160, 176]

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_3(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)

def _perform_ut(data_path:str=None):
    if not data_path:
        data_path = "tests/data/ames_house_prices_data.csv"
    df = pd.read_csv(data_path)
    df_train = df.iloc[:-50, :]
    df_test = df.iloc[-50:, :]

    target_col = 'SalePrice'
    feats_in_asc_order = ['1stFlrSF', 'TotalBsmtSF', 'GrLivArea']
    feats_in_no_order = []

    min_gap_pct = 0.5
    lb = 0.0
    ub = 100.0

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_3(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)
    
def _perform_st_trend(data_path:str=None):
    if not data_path:
        data_path = "tests/data/trend_data.csv"
    df = pd.read_csv(data_path)
    df_train = df.iloc[:12, :]
    df_test = df.iloc[12:, :]

    target_col = 'Cost'
    feats_in_asc_order = ['HC1', 'HC2', 'HC3']
    feats_in_no_order = ['Idx']

    min_gap_pct = [0.5, 0.1]
    lb = 10.0
    ub = [100.0, 100.0, 1e3, np.inf]

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_3(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)

def _perform_st_scb_swe_male_non_manual_pvt_wages(data_path:str=None):
    if not data_path:
        data_path = "tests/data/scb_swe_male_non_manual_pvt_wages_data.xlsx"
    df = pd.read_excel(data_path, sheet_name="data")
    df_train = df.iloc[:3, :]
    df_test = df.iloc[3:, :]

    target_col = 'Cost'
    feats_in_asc_order = ['HC3', 'HC2', 'HC1']
    feats_in_no_order = []

    min_gap_pct = 0.1
    lb = 40000
    ub = 100000

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_3(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)

if __name__ == '__main__':
    msg = "CROMP testing pipeline." + "\nUse -b option to select benchmarking test with n number of training samples." +\
          "\nUse -s option to select system test" +\
          "(default mode: trend, modes supported: scb_swe_male_non_manual_pvt_wages, trend)." +\
          "\nUse -u option to select unit test."
    parser = argparse.ArgumentParser(description=msg)
    
    parser.add_argument("-b", "--Benchmark", default=False, action="store_true",\
                        help="select benchmarking test")
    parser.add_argument("NumTrainSamples", metavar='n', type=int, nargs='?',\
                        help="specify number of training samples for benchmarking test")
    
    parser.add_argument("-s", "--SystemTest", metavar='mode', type=str, nargs='?', const='trend_data',\
                        help="select system test (default mode: trend_data, modes supported: trend_data)")
    
    parser.add_argument("-u", "--UT", default=False, action="store_true",\
                        help="select unit test")
    
    args = parser.parse_args()
    
    if args.Benchmark:
        if args.NumTrainSamples:
            print("\nBenchmarking test with {} training samples:".format(args.NumTrainSamples))
            print("\n===========================================\n")
            _perform_benchmarking(num_training_samples=args.NumTrainSamples)
        else:
            print("\nERROR! Benchmarking test requires number of training samples to be specified.\n")
    elif args.SystemTest:
        print("\nSystem test with {} mode:".format(args.SystemTest))
        print("\n=========================\n")
        if args.SystemTest == 'scb_swe_male_non_manual_pvt_wages':
            _perform_st_scb_swe_male_non_manual_pvt_wages()
        elif args.SystemTest == 'trend':
            _perform_st_trend()
        else:
            print("\nERROR! Unsupported mode specified for system test." +\
                  "Modes supported are: scb_swe_male_non_manual_pvt_wages, trend.\n")
    elif args.UT:
        print("\nUnit test:")
        print("\n==========\n")
        _perform_ut()
    else:
        print(msg)

        # For debugging purposes only
        #_perform_ut(data_path="data/ames_house_prices_data.csv")
        #_perform_benchmarking(num_training_samples=12, data_path="data/benchmark_wage_data.xlsx")
        #_perform_st_scb_swe_male_non_manual_pvt_wages(data_path="data/scb_swe_male_non_manual_pvt_wages_data.xlsx")
        _perform_st_trend(data_path="data/trend_data.csv")

    gc.collect()
    
