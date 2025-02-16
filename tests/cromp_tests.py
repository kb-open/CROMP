# CROMP testing pipeline
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

import gc, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import lsq_linear
import cvxpy as cp
import pymc as pm
from cromp import CROMPTrain, CROMPPredict

def _mape(predictions, actuals):
    return abs((predictions - actuals) / actuals).mean()

def _test_1(df_train, df_test, target_col,
            feats_in_asc_order, min_gap_pct,
            feats_in_no_order,
            lb, ub,
            no_intercept=True):
    model = CROMPTrain()
    ret_success = model.config_constraints(feats_in_asc_order, min_gap_pct=min_gap_pct,\
                                           feats_in_no_order=feats_in_no_order,\
                                           lb=lb, ub=ub,
                                           no_intercept=no_intercept)
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

def _test_2(df_train, df_test, target_col, feats):
    model = LinearRegression()
    model.fit(df_train[feats], df_train[target_col])
    print("\nPredicted intercept from MLR:", model.intercept_)
    print("Predicted coefficients from MLR:", model.coef_)

    result = model.predict(df_test[feats])
    print("Predicted result from MLR:\n", result)
    print("MAPE from MLR:", _mape(result, df_test[target_col]))

def _test_3(df_train, df_test, target_col, feats):
    model = LinearRegression(fit_intercept=False)
    model.fit(df_train[feats], df_train[target_col])
    print("\nPredicted coefficients from MLR without intercept:", model.coef_)

    result = model.predict(df_test[feats])
    print("Predicted result from MLR without intercept:\n", result)
    print("MAPE from MLR without intercept:", _mape(result, df_test[target_col]))

def _test_4(df_train, df_test, target_col, feats, lb, ub):
    model = lsq_linear(df_train[feats], df_train[target_col], bounds=(lb, ub))
    print("\nPredicted coefficients from Linear LSQ:", model.x)

    result = df_test.apply(lambda row: sum([x * y for x, y in zip(model.x, row[feats])]),
                           axis=1)
    print("Predicted result from Linear LSQ:\n", result)
    print("MAPE from Linear LSQ:", _mape(result, df_test[target_col]))

def _test_5(df_train, df_test, target_col, feats, lb, ub):
    n_coef = len(feats)
    coef = cp.Variable(n_coef)
    constraints = ([coef[i] >= lb[i] for i in range(n_coef) if lb[i] is not None] +\
                               [coef[i] <= ub[i] for i in range(n_coef) if ub[i] is not None])
    objective = cp.Minimize(cp.sum_squares(df_train[feats].values @ coef - df_train[target_col].values))
    model = cp.Problem(objective, constraints)
    model.solve()
    if model.status == 'optimal':
        print("\nPredicted coefficients from CVXPY:", coef.value)
        
        result = df_test[feats].values @ coef.value
        print("Predicted result from CVXPY:\n", result)
        print("MAPE from CVXPY:", _mape(result, df_test[target_col]))

def _test_6(df_train, df_test, target_col, feats, lb, ub):
    coords = {'features': feats}
    coords_mutable = {'obs_id': np.arange(len(df_train)).tolist()}
    
    print("\nBayesian")
    with pm.Model(coords=coords, coords_mutable=coords_mutable) as model:
        X = pm.MutableData("X", df_train[feats], dims=('obs_id', 'features'))
        std = df_train[target_col].std()
        
        intercept = 0 #pm.Normal("intercept", mu=0, sigma=std)
        beta = pm.TruncatedNormal("beta", mu=0, sigma=std, lower=lb, upper=ub, shape=len(feats), dims='features')
        sigma = pm.HalfNormal("sigma", sigma=std)
        
        mu = pm.Deterministic("mu", intercept + pm.math.dot(X, beta), dims='obs_id')
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=df_train[target_col], dims='obs_id')
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.85, random_seed=1)
    
    divergences = trace.sample_stats["diverging"].sum().item()
    if not divergences:
        beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw"))
        print("\nPredicted coefficients from Bayesian:", beta_mean.values)
        
        with model:
            pm.set_data({"X": df_test[feats]},
                        coords={'obs_id': [i+len(df_train) for i in np.arange(len(df_test)).tolist()]})
            posterior_predictive = pm.sample_posterior_predictive(trace, predictions=True)
            result = posterior_predictive["predictions"]["y"].mean(dim=("chain", "draw")).values.flatten()
            print("Predicted result from Bayesian:\n", result)
            print("MAPE from Bayesian:", _mape(result, df_test[target_col]))

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
    _test_5(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)
    _test_6(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)

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
    _test_5(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)
    _test_6(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)
    
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
    _test_5(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)
    _test_6(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)

def _perform_st_scb_swe_male_non_manual_pvt_wages(data_path:str=None):
    if not data_path:
        data_path = "tests/data/scb_swe_male_non_manual_pvt_wages_data.xlsx"
    df = pd.read_excel(data_path, sheet_name="data")
    df_train = df.iloc[:4, :]
    df_test = df.iloc[4:, :]

    target_col = 'Cost'
    feats_in_asc_order = ['HC1', 'HC2', 'HC3', 'HC4']
    feats_in_no_order = ['Year']

    min_gap_pct = [0.25, 0.07, 0.07]
    lb = [40000, 40000, 40000, 40000, 0]
    ub = [100000, 100000, 100000, 100000, np.Inf]

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_3(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)
    _test_5(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)
    _test_6(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub)

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
    
    parser.add_argument("-s", "--SystemTest", metavar='mode', type=str, nargs='?', const='scb',\
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
        if args.SystemTest == 'scb':
            _perform_st_scb_swe_male_non_manual_pvt_wages()
        elif args.SystemTest == 'trend':
            _perform_st_trend()
        else:
            print("\nERROR! Unsupported mode specified for system test." +\
                  "Modes supported are: scb, trend.\n")
    elif args.UT:
        print("\nUnit test:")
        print("\n==========\n")
        _perform_ut()
    else:
        print(msg)

        # For debugging purposes only
        #_perform_ut(data_path="data/ames_house_prices_data.csv")
        #_perform_benchmarking(num_training_samples=3, data_path="data/benchmark_wage_data.xlsx")
        #_perform_st_scb_swe_male_non_manual_pvt_wages(data_path="data/scb_swe_male_non_manual_pvt_wages_data.xlsx")
        #_perform_st_trend(data_path="data/trend_data.csv")

    gc.collect()
    
