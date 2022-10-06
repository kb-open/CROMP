# Implementation of CROMP training
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

configs = {'low_val': 0.0001, 'high_val': np.inf}

class CROMPTrain():
    def __init__(self, verbose:bool=False):
        self.verbose = verbose

    def configure(self, df:pd.DataFrame, target_col:str, feature_cols_in_asc_order:[str],
                  min_gap_pct:[float]=0.0, lb:[float]=0.0, ub:[float]=configs['high_val'],
                  no_intercept:bool=False) -> bool:
        self.df = df
        self.target = target_col
        self.features = feature_cols_in_asc_order

        # Initialize coefficients
        self.len_feats = len(self.features)
        self.len_coeffs = self.len_feats + 1
        self.coeffs = list(np.zeros(self.len_coeffs))

        # Initialize lb and ub constraints
        min_con_orig = list(np.zeros(self.len_coeffs))
        if isinstance(lb, list):
            if len(lb) == self.len_coeffs:
                min_con_orig = lb
            elif len(lb) == self.len_coeffs - 1:
                min_con_orig[1:] = lb
            else:
                print("INCORRECT CONFIG: Number of lower bounds do not match with number of features passed!")
                return False
        elif lb != 0.0:
            min_con_orig[1:] = [lb for idx, _ in enumerate(min_con_orig) if idx > 0]

        max_con_orig = [configs['high_val'] for i in range(self.len_coeffs)]
        if isinstance(ub, list):
            if len(ub) == self.len_coeffs:
                max_con_orig = ub
            elif len(ub) == self.len_coeffs - 1:
                max_con_orig[1:] = ub
            else:
                print("INCORRECT CONFIG: Number of upper bounds do not match with number of features passed!")
                return False
        elif ub != configs['high_val']:
            max_con_orig[1:] = [ub for idx, _ in enumerate(max_con_orig) if idx > 0]

        if no_intercept:
            min_con_orig[0] = 0.0
            max_con_orig[0] = configs['low_val']

        # Configure constraints
        if not self._min_gap_pct_constraint(min_gap_pct):
            return False

        if not self._lb_ub_constraint(min_con_orig, max_con_orig):
            return False

        if self.verbose:
            print("Initial coefficients:", self.coeffs)
            print("Minimum gap percentages:", self.min_gap_pct)
            print("Minimum constraints:", self.min_con)
            print("Maximum constraints:", self.max_con)

        return True

    def train(self) -> (bool, [float]):
        # Build features amenable for CROMP
        X = self._feat_eng()

        # Convert independent variables to a matrix
        X = X.values

        # Add an array of ones to act as intercept coefficient
        ones = np.ones(X.shape[0])
        # Combine array of ones and indepedent variables
        X = np.concatenate((ones[:, np.newaxis], X), axis=1)

        # Convert target variable to a matrix
        y = self.df[self.target].values

        # Run optimization
        results = lsq_linear(X, y, bounds=(self.min_con, self.max_con), lsmr_tol='auto')
        if self.verbose:
            print("\nResults:\n", results)

        if results.success:
            # Get the coefficients back to the space of original features
            self._get_coeff(results)
            if self.verbose:
                print("\nFinal Coefficients (including intercept):", self.coeffs)
        else:
            print("ERROR: Convergence was not achieved!")

        return results.success, self.coeffs

    def _min_gap_pct_constraint(self, min_gap_pct:[float]) -> bool:
        self.min_gap_pct = list(np.zeros(self.len_feats))

        if isinstance(min_gap_pct, list):
            if len(min_gap_pct) == self.len_feats:
                self.min_gap_pct[1:] = min_gap_pct[1:]
            elif len(min_gap_pct) == self.len_feats - 1:
                self.min_gap_pct[1:] = min_gap_pct
            else:
                print("INCORRECT CONFIG: Number of percentage gaps do not match with number of features passed!")
                return False
        elif min_gap_pct != 0.0:
            self.min_gap_pct[1:] = [min_gap_pct for idx, _ in enumerate(self.min_gap_pct) if idx > 0]

        if min(self.min_gap_pct) < 0.0:
            print("INCORRECT CONFIG: Percentage gaps cannot be negative!")
            return False

        return True

    def _lb_ub_constraint(self, min_con_orig:[float], max_con_orig:[float]) -> bool:
        self.min_con = min_con_orig.copy()
        self.max_con = max_con_orig.copy()

        for i in range(2, self.len_coeffs):
            self.min_con[i] = max(self.min_con[i] - (1 + self.min_gap_pct[i - 1]) * min_con_orig[i - 1],
                                  self.min_con[i] - (1 + self.min_gap_pct[i - 1]) * max_con_orig[i - 1])

            self.max_con[i] = min(self.max_con[i] - (1 + self.min_gap_pct[i - 1]) * min_con_orig[i - 1],
                                  self.max_con[i] - (1 + self.min_gap_pct[i - 1]) * max_con_orig[i - 1])

        for i in range(self.len_coeffs):
            self.min_con[i] = max(0, self.min_con[i])
            self.max_con[i] = max(self.min_con[i] + configs['low_val'], self.max_con[i])

        return True

    def _feat_eng(self) -> pd.DataFrame:
        X = self.df[self.features].copy()
        tmp = X.copy()

        i = self.len_feats
        tmp[f'F{i}'] = tmp[self.features[i - 1]]

        while (i - 1):
            tmp[f'F{i-1}'] = (1 + self.min_gap_pct[i - 1]) * tmp[f'F{i}'] + tmp[self.features[i - 2]]
            i -= 1

        for i in range(self.len_feats):
            X[f'F{i+1}'] = tmp[f'F{i+1}']

        X = X.drop(self.features, axis=1)
        del tmp

        return X

    def _get_coeff(self, results:dict):
        self.coeffs[0] = results.x[0]
        self.coeffs[1] = results.x[1]
        for i in range(self.len_coeffs - 2):
            self.coeffs[i + 2] = (1 + self.min_gap_pct[i + 1]) * self.coeffs[i + 1] + results.x[i + 2]

if __name__ == '__main__':
    pass
    
