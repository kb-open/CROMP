# CROMP testing pipeline
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

import gc, argparse, time
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from scipy.optimize import lsq_linear
import cvxpy as cp
import arviz as az
import pymc as pm
from cromp import CROMPTrain, CROMPPredict

def _mape(predictions, actuals):
    return abs((predictions - actuals) / actuals).mean()

def _test_1(df_train, df_test, target_col,
            feats_in_asc_order, min_gap_pct,
            feats_in_no_order,
            lb, ub,
            no_intercept=True, cnf_int=True, verbosity=2):
    start_time = time.time()
    model = CROMPTrain()
    ret_success = model.config_constraints(feats_in_asc_order, min_gap_pct=min_gap_pct,\
                                           feats_in_no_order=feats_in_no_order,\
                                           lb=lb, ub=ub,
                                           no_intercept=no_intercept)
    if ret_success:
        ret_success, cromp_model = model.train(df_train, target_col)
    print("\n***Time taken by CROMP training:", time.time() - start_time, "seconds.")

    if ret_success:
        model = CROMPPredict(cromp_model)
        
        start_time = time.time()
        result = model.predict(df_test)
        print("***Time taken by CROMP prediction:", time.time() - start_time, "seconds.")
        
        if ret_success:
            if cnf_int:
                print(">>>Bootstrapped confidence interval computation begins>>>")
                intercept_db = []
                coefs_db = []
                result_db = []
                for _ in range(1000):
                    df_b = resample(df_train)
                    model_b = CROMPTrain()
                    ret_success = model_b.config_constraints(feats_in_asc_order, min_gap_pct=min_gap_pct,\
                                                             feats_in_no_order=feats_in_no_order,\
                                                             lb=lb, ub=ub,
                                                             no_intercept=no_intercept)
                    if ret_success:
                        ret_success, cromp_model_b = model_b.train(df_b, target_col)
                    if ret_success:
                        model_b = CROMPPredict(cromp_model_b)
                    intercept_db.append(cromp_model_b['coeffs'][0])
                    coefs_db.append(cromp_model_b['coeffs'][1:])
                    result_db.append(model_b.predict(df_test))
                intercept_db = np.array(intercept_db)
                intercept_ci = np.percentile(intercept_db, [2.5, 97.5])
                coefs_db = np.array(coefs_db)
                coef_ci = np.percentile(coefs_db, [2.5, 97.5], axis=0)
                coef_ci = [list(item) for item in zip(coef_ci[0], coef_ci[1])]
                result_db = np.array(result_db)
                result_ci = np.percentile(result_db, [2.5, 97.5], axis=0)
                result_ci = [list(item) for item in zip(result_ci[0], result_ci[1])]
                print("<<<Bootstrapped confidence interval computation ends<<<")
        
            print("\nPredicted intercept from CROMP:", cromp_model['coeffs'][0])
            if cnf_int:
                print("95% Confidence Interval (Bootstrapped) for intercept from CROMP:\n", intercept_ci)
                
            print("Predicted coefficients from CROMP:", cromp_model['coeffs'][1:])
            if cnf_int:
                print("95% Confidence Interval (Bootstrapped) for coefficients from CROMP:\n", coef_ci)
            
            if verbosity >= 2:
                print("Predicted result from CROMP:\n", result)
                if cnf_int:
                    print("95% Confidence Intervals (Bootstrapped) for predicted result from CROMP:\n", result_ci)
                
            print("MAPE from CROMP:", _mape(result, df_test[target_col]))

def _test_2(df_train, df_test, target_col, feats, cnf_int=True, verbosity=2):
    start_time = time.time()
    model = LinearRegression()
    model.fit(df_train[feats], df_train[target_col])
    print("\n***Time taken by MLR training:", time.time() - start_time, "seconds.")
    
    start_time = time.time()
    result = model.predict(df_test[feats])
    print("***Time taken by MLR prediction:", time.time() - start_time, "seconds.")
    
    if cnf_int:
        print(">>>Bootstrapped confidence interval computation begins>>>")
        intercept_db = []
        coefs_db = []
        result_db = []
        for _ in range(1000):
            df_b = resample(df_train)
            model_b = LinearRegression()
            model_b.fit(df_b[feats], df_b[target_col])
            intercept_db.append(model_b.intercept_)
            coefs_db.append(model_b.coef_)
            result_db.append(model_b.predict(df_test[feats]))
        intercept_db = np.array(intercept_db)
        intercept_ci = np.percentile(intercept_db, [2.5, 97.5])
        coefs_db = np.array(coefs_db)
        coef_ci = np.percentile(coefs_db, [2.5, 97.5], axis=0)
        coef_ci = [list(item) for item in zip(coef_ci[0], coef_ci[1])]
        result_db = np.array(result_db)
        result_ci = np.percentile(result_db, [2.5, 97.5], axis=0)
        result_ci = [list(item) for item in zip(result_ci[0], result_ci[1])]
        print("<<<Bootstrapped confidence interval computation ends<<<")
    
    print("\nPredicted intercept from MLR:", model.intercept_)
    if cnf_int:
        print("95% Confidence Interval (Bootstrapped) for intercept from MLR:\n", intercept_ci)
        
    print("Predicted coefficients from MLR:", model.coef_)
    if cnf_int:
        print("95% Confidence Interval (Bootstrapped) for coefficients from MLR:\n", coef_ci)
    
    if verbosity >= 2:
        print("Predicted result from MLR:\n", result)
        if cnf_int:
            print("95% Confidence Intervals (Bootstrapped) for predicted result from MLR:\n", result_ci)
        
    print("MAPE from MLR:", _mape(result, df_test[target_col]))

def _test_3(df_train, df_test, target_col, feats, cnf_int=True, verbosity=2):
    start_time = time.time()
    model = LinearRegression(fit_intercept=False)
    model.fit(df_train[feats], df_train[target_col])
    print("\n***Time taken by MLR without intercept training:", time.time() - start_time, "seconds.")
    
    start_time = time.time()
    result = model.predict(df_test[feats])
    print("***Time taken by MLR without intercept prediction:", time.time() - start_time, "seconds.")
    
    if cnf_int:
        print(">>>Bootstrapped confidence interval computation begins>>>")
        coefs_db = []
        result_db = []
        for _ in range(1000):
            df_b = resample(df_train)
            model_b = LinearRegression(fit_intercept=False)
            model_b.fit(df_b[feats], df_b[target_col])
            coefs_db.append(model_b.coef_)
            result_db.append(model_b.predict(df_test[feats]))
        coefs_db = np.array(coefs_db)
        coef_ci = np.percentile(coefs_db, [2.5, 97.5], axis=0)
        coef_ci = [list(item) for item in zip(coef_ci[0], coef_ci[1])]
        result_db = np.array(result_db)
        result_ci = np.percentile(result_db, [2.5, 97.5], axis=0)
        result_ci = [list(item) for item in zip(result_ci[0], result_ci[1])]
        print("<<<Bootstrapped confidence interval computation ends<<<")
        
    print("\nPredicted coefficients from MLR without intercept:", model.coef_)
    if cnf_int:
        print("95% Confidence Interval (Bootstrapped) for coefficients from MLR without intercept:\n", coef_ci)
    
    if verbosity >= 2:    
        print("Predicted result from MLR without intercept:\n", result)
        if cnf_int:
            print("95% Confidence Intervals (Bootstrapped) for predicted result from MLR without intercept:\n", result_ci)
        
    print("MAPE from MLR without intercept:", _mape(result, df_test[target_col]))

def _test_4(df_train, df_test, target_col, feats, lb, ub, cnf_int=True, verbosity=2):
    start_time = time.time()
    model = lsq_linear(df_train[feats], df_train[target_col], bounds=(lb, ub))
    print("\n***Time taken by Linear LSQ training:", time.time() - start_time, "seconds.")
    
    start_time = time.time()
    result = df_test.apply(lambda row: sum([x * y for x, y in zip(model.x, row[feats])]), axis=1)
    print("***Time taken by Linear LSQ prediction:", time.time() - start_time, "seconds.")
    
    if cnf_int:
        print(">>>Bootstrapped confidence interval computation begins>>>")
        coefs_db = []
        result_db = []
        for _ in range(1000):
            df_b = resample(df_train)
            try:
                model_b = lsq_linear(df_b[feats], df_b[target_col], bounds=(lb, ub))
                if model_b.success:
                    coefs_db.append(model_b.x)
                    result_db.append(df_test[feats] @ model_b.x)
            except:
                continue  # Skip failed iterations
        coefs_db = np.array(coefs_db)
        coef_ci = np.percentile(coefs_db, [2.5, 97.5], axis=0)
        coef_ci = [list(item) for item in zip(coef_ci[0], coef_ci[1])]
        result_db = np.array(result_db)
        result_ci = np.percentile(result_db, [2.5, 97.5], axis=0)
        result_ci = [list(item) for item in zip(result_ci[0], result_ci[1])]
        print("<<<Bootstrapped confidence interval computation ends<<<")
    
    print("\nPredicted coefficients from Linear LSQ:", model.x)
    if cnf_int:
        print("95% Confidence Interval (Bootstrapped) for coefficients from Linear LSQ:\n", coef_ci)
    
    if verbosity >= 2:
        print("Predicted result from Linear LSQ:\n", result)
        if cnf_int:
            print("95% Confidence Intervals (Bootstrapped) for predicted result from Linear LSQ:\n", result_ci)
        
    print("MAPE from Linear LSQ:", _mape(result, df_test[target_col]))

def _test_5(df_train, df_test, target_col, feats, lb, ub, cnf_int=True, verbosity=2):
    start_time = time.time()
    n_coef = len(feats)
    
    if lb is None:
        lb = [None] * n_coef
    elif not isinstance(lb, (list, tuple)):
        lb = [lb] * n_coef

    if ub is None:
        ub = [None] * n_coef
    elif not isinstance(ub, (list, tuple)):
        ub = [ub] * n_coef
    
    coef = cp.Variable(n_coef)    
    constraints = ([coef[i] >= lb[i] for i in range(n_coef) if lb[i] is not None] +\
                   [coef[i] <= ub[i] for i in range(n_coef) if ub[i] is not None])
                   
    objective = cp.Minimize(cp.sum_squares(df_train[feats].values @ coef - df_train[target_col].values))
    model = cp.Problem(objective, constraints)
    model.solve()
    print("\n***Time taken by CVXPY training:", time.time() - start_time, "seconds.")
    
    if model.status == 'optimal':  
        start_time = time.time()
        result = df_test[feats].values @ coef.value
        print("***Time taken by CVXPY prediction:", time.time() - start_time, "seconds.")
        
        if cnf_int:
            print(">>>Bootstrapped confidence interval computation begins>>>")
            coefs_db = []
            result_db = []
            for _ in range(1000):
                df_b = resample(df_train)
                coef_b = cp.Variable(n_coef)
                constraints_b = ([coef_b[i] >= lb[i] for i in range(n_coef) if lb[i] is not None] +
                                 [coef_b[i] <= ub[i] for i in range(n_coef) if ub[i] is not None])
                objective_b = cp.Minimize(cp.sum_squares(df_b[feats].values @ coef_b - df_b[target_col].values))
                model_b = cp.Problem(objective_b, constraints_b)
                try:
                    model_b.solve(solver=cp.OSQP, warm_start=True)
                    if model_b.status == 'optimal':
                        coefs_db.append(coef_b.value)
                        result_db.append(df_test[feats].values @ coef_b.value)
                except:
                    continue  # Skip failed iterations
            coefs_db = np.array(coefs_db)
            coef_ci = np.percentile(coefs_db, [2.5, 97.5], axis=0)
            coef_ci = [list(item) for item in zip(coef_ci[0], coef_ci[1])]
            result_db = np.array(result_db)
            result_ci = np.percentile(result_db, [2.5, 97.5], axis=0)
            result_ci = [list(item) for item in zip(result_ci[0], result_ci[1])]
            print("<<<Bootstrapped confidence interval computation ends<<<")
            
        print("\nPredicted coefficients from CVXPY:", coef.value)
        if cnf_int:
            print("95% Confidence Interval (Bootstrapped) for coefficients from CVXPY:\n", coef_ci)
        
        if verbosity >= 2:        
            print("Predicted result from CVXPY:\n", result)
            if cnf_int:
                print("95% Confidence Intervals (Bootstrapped) for predicted result from CVXPY:\n", result_ci)
            
        print("MAPE from CVXPY:", _mape(result, df_test[target_col]))
        
def _test_6(df_train, df_test, target_col, feats, lb, ub, no_intercept=True, cnf_int=True, verbosity=2):
    print("\n>>>Bayesian begins>>>")
    start_time = time.time()
    coords = {'features': feats}
    coords_mutable = {'obs_id': np.arange(len(df_train)).tolist()}
    
    with pm.Model(coords=coords, coords_mutable=coords_mutable) as model:
        X = pm.MutableData("X", df_train[feats], dims=('obs_id', 'features'))

        std = df_train[target_col].std()
        
        if no_intercept: 
            intercept = pm.HalfNormal("intercept", sigma=1) #pm.Normal("intercept", mu=0, sigma=1)
        else:
            intercept = pm.HalfNormal("intercept", sigma=std) #pm.Normal("intercept", mu=0, sigma=std)
        beta = pm.TruncatedNormal("beta", mu=0, sigma=std, lower=lb, upper=ub, shape=len(feats), dims='features')
        sigma = pm.HalfNormal("sigma", sigma=std)
        
        mu = pm.Deterministic("mu", intercept + pm.math.dot(X, beta), dims='obs_id')
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=df_train[target_col], dims='obs_id')
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.85, random_seed=1)
    
    divergences = trace.sample_stats["diverging"].sum().item()
    print("\n***Time taken by Bayesian training:", time.time() - start_time, "seconds.")
    
    if not divergences:
        if "intercept" in trace.posterior:
            print("\nPredicted intercept from Bayesian:", trace.posterior["intercept"].mean(dim=("chain", "draw")).values)
            
            if cnf_int:
                intercept_ci = az.hdi(trace.posterior["intercept"], hdi_prob=0.95)["intercept"].values
                print("95% Credible Interval for intercept from Bayesian:\n", intercept_ci)
        else:
            print("\nUsed intercept in Bayesian:", intercept)
        
        beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw"))
        print("Predicted coefficients from Bayesian:", trace.posterior["beta"].mean(dim=("chain", "draw")).values)
        
        if cnf_int:
            beta_ci = az.hdi(trace.posterior["beta"], hdi_prob=0.95)["beta"].values
            print("95% Credible Interval for coefficients from Bayesian:\n", beta_ci)
        
        with model:
            start_time = time.time()
            pm.set_data({"X": df_test[feats]},
                        coords={'obs_id': [i+len(df_train) for i in np.arange(len(df_test)).tolist()]})
            posterior_predictive = pm.sample_posterior_predictive(trace, predictions=True)
            result = posterior_predictive["predictions"]["y"].mean(dim=("chain", "draw")).values.flatten()
            print("\n***Time taken by Bayesian prediction:", time.time() - start_time, "seconds.")
            
            if verbosity >= 2:
                print("\nPredicted result from Bayesian:\n", result)
                if cnf_int:
                    result_samples = posterior_predictive["predictions"]["y"].stack(sample=("chain", "draw")).values
                    result_ci = [list(item) for item in zip(np.percentile(result_samples, 2.5, axis=1), np.percentile(result_samples, 97.5, axis=1))]
                    print("95% Credible Intervals for predicted result from Bayesian:\n", result_ci)
            
            print("MAPE from Bayesian:", _mape(result, df_test[target_col]))
    print("<<<Bayesian ends<<<")
            
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

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub, cnf_int=False)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, cnf_int=False)
    _test_3(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, cnf_int=False)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, cnf_int=False)
    _test_5(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, cnf_int=False)
    _test_6(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, cnf_int=False)

def _perform_mt(data_path:str=None):
    if not data_path:
        data_path = "tests/data/Advertising.csv"
    df = pd.read_csv(data_path)
    
    df['revenue'] = df['sales']*100 # For meaningful ROI bounds typically found in the industry
        
    df_train = df.iloc[:-50, :]
    df_test = df.iloc[-50:, :]

    target_col = 'revenue'
    feats_in_asc_order = ['TV', 'newspaper', 'radio']
    feats_in_no_order = []

    min_gap_pct = 0
    lb = [0.5, 1.36, 9]
    ub = [6.16, 4.01, 25]

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub, no_intercept=False, verbosity=1)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, verbosity=1)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, verbosity=1)
    _test_5(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, verbosity=1)
    _test_6(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, no_intercept=False, verbosity=1)

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

    _test_1(df_train, df_test, target_col, feats_in_asc_order, min_gap_pct, feats_in_no_order, lb, ub, cnf_int=False)
    _test_2(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, cnf_int=False)
    _test_3(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, cnf_int=False)
    _test_4(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, cnf_int=False)
    _test_5(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, cnf_int=False)
    _test_6(df_train, df_test, target_col, feats_in_asc_order + feats_in_no_order, lb, ub, cnf_int=False)

if __name__ == '__main__':
    msg = "CROMP testing pipeline." +\
          "\nUse -b option to select benchmarking test with n number of training samples." +\
          "\nUse -s option to select system test" + "(default mode: trend, modes supported: scb_swe_male_non_manual_pvt_wages, trend)." +\
          "\nUse -m option to select module test." +\
          "\nUse -u option to select unit test."
    parser = argparse.ArgumentParser(description=msg)
    
    parser.add_argument("-b", "--Benchmark", default=False, action="store_true",\
                        help="select benchmarking test")
    parser.add_argument("NumTrainSamples", metavar='n', type=int, nargs='?',\
                        help="specify number of training samples for benchmarking test")
    
    parser.add_argument("-s", "--SystemTest", metavar='mode', type=str, nargs='?', const='scb',\
                        help="select system test (default mode: trend_data, modes supported: trend_data)")
    
    parser.add_argument("-m", "--ModuleTest", default=False, action="store_true",\
                        help="select module test")
    
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
    elif args.ModuleTest:
        print("\nModule test:")
        print("\n============\n")
        _perform_mt()
    elif args.UT:
        print("\nUnit test:")
        print("\n==========\n")
        _perform_ut()
    else:
        print(msg)

        # For debugging purposes only
        #_perform_ut(data_path="data/ames_house_prices_data.csv")
        #_perform_mt(data_path="data/Advertising.csv")
        #_perform_benchmarking(num_training_samples=3, data_path="data/benchmark_wage_data.xlsx")
        #_perform_st_scb_swe_male_non_manual_pvt_wages(data_path="data/scb_swe_male_non_manual_pvt_wages_data.xlsx")
        #_perform_st_trend(data_path="data/trend_data.csv")

    gc.collect()
    
