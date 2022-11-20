# Implementation of CROMP training
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

from joblib import dump
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

configs = {'low_val': 1e-9, 'high_val': 1e9, 'tol': 1e-10}

class CROMPTrain():
    def __init__(self, verbose:bool=False):
        self.verbose = verbose

    def config_constraints(self, feats_in_asc_order:[str], min_gap_pct:[float]=0.0,
                           feats_in_no_order:[str]=[],
                           lb:[float]=configs['low_val'], ub:[float]=configs['high_val'],
                           no_intercept:bool=True) -> bool:
        self.feats_in_asc_order = feats_in_asc_order
        self.feats_in_no_order = feats_in_no_order

        # Initialize coefficients
        self.len_feats_in_asc_order = len(self.feats_in_asc_order)
        self.len_feats_in_no_order = len(self.feats_in_no_order)
        self.len_coeffs = 1 + self.len_feats_in_asc_order + self.len_feats_in_no_order
        self.coeffs = list(np.zeros(self.len_coeffs))

        # Configure margin constraints
        if not self._min_gap_pct_constraints(min_gap_pct):
            return False

        # Configure bound constraints
        if not self._bound_constraints(lb, ub, no_intercept):
            return False

        if self.verbose:
            print("Initial coefficients:", self.coeffs)
            print("Minimum gap percentages:", self.min_gap_pct)
            print("Minimum constraints:", self.min_con)
            print("Maximum constraints:", self.max_con)

        return True

    def train(self, df:pd.DataFrame, target_col:str) -> (bool, dict):
        self.target = target_col

        self.cromp_model = {}
        self.iter_data = {}

        for self.anchor_coeff_idx in range(self.len_feats_in_asc_order + 1):
            self.iter_data[self.anchor_coeff_idx] = {}

            # Finalize bound constraints
            self._finalize_bound_constraints()

            # Build feats_in_asc_order amenable for CROMP
            X = self._feat_eng(df)

            # Convert target variable to a matrix
            y = df[self.target].values

            # Run optimization
            results = lsq_linear(X, y, bounds=(self.min_con, self.max_con), lsmr_tol='auto')
            if self.verbose:
                print("\nAnchor coefficient {}: Results: {}\n".format(self.anchor_coeff_idx, results))

            if results.success:
                # Get the coefficients back to the space of original feats_in_asc_order
                if self._get_coeff(results):
                    self.iter_data[self.anchor_coeff_idx]['success'] = True
                    if self.verbose:
                        print("\nAnchor coefficient {}: Coefficients (including intercept):".format(self.anchor_coeff_idx,
                                                                                                   self.coeffs))
                else:
                    self.iter_data[self.anchor_coeff_idx]['success'] = False
            else:
                self.iter_data[self.anchor_coeff_idx]['success'] = False

            if self.iter_data[self.anchor_coeff_idx]['success']:
                # Save results
                self.iter_data[self.anchor_coeff_idx]['results'] = results.copy()
                self.iter_data[self.anchor_coeff_idx]['coeffs'] = self.coeffs.copy()

        # Use best iteration
        self.best_anchor_coeff_idx = 0
        success = False
        for anchor_coeff_idx in range(self.len_feats_in_asc_order + 1):
            if self.iter_data[anchor_coeff_idx]['success']:
                if not success:
                    success = True
                    self.best_anchor_coeff_idx = anchor_coeff_idx
                elif self.iter_data[anchor_coeff_idx]['results']['cost'] < self.iter_data[self.best_anchor_coeff_idx]['results']['cost']:
                    self.best_anchor_coeff_idx = anchor_coeff_idx

        if not success:
            print("ERROR: Convergence was not achieved!")
        else:
            self.cromp_model['coeffs'] = self.iter_data[self.best_anchor_coeff_idx]['coeffs']
            self.cromp_model['feats'] =  self.feats_in_asc_order + self.feats_in_no_order

        return success, self.cromp_model

    def save(self, path:str):
        dump(self.cromp_model, path.rstrip('/') + '/cromp_model.joblib')

    def _min_gap_pct_constraints(self, min_gap_pct:[float]) -> bool:
        self.min_gap_pct = list(np.zeros(self.len_feats_in_asc_order))

        if isinstance(min_gap_pct, list):
            if len(min_gap_pct) == self.len_feats_in_asc_order - 1:
                self.min_gap_pct[1:] = min_gap_pct
            else:
                print("INCORRECT CONFIG: Length of percentage gaps do not match with length of feats_in_asc_order passed!")
                print("Expected: {}, Received: {}".format(self.len_feats_in_asc_order - 1, len(min_gap_pct)))
                return False
        elif min_gap_pct != 0.0:
            self.min_gap_pct[1:] = [min_gap_pct for idx, _ in enumerate(self.min_gap_pct) if idx > 0]

        if min(self.min_gap_pct) < 0.0:
            print("INCORRECT CONFIG: Percentage gaps cannot be negative!")
            return False

        return True

    def _bound_constraints(self, lb:[float], ub:[float], no_intercept:bool) -> bool:
        # Initialize lb constraints
        min_con_orig = [configs['low_val'] for i in range(self.len_coeffs)]
        if isinstance(lb, list):
            if len(lb) == self.len_coeffs:
                min_con_orig = lb
            elif len(lb) == self.len_coeffs - 1:
                min_con_orig[1:] = lb
            else:
                print("INCORRECT CONFIG: Number of lower bounds do not match with lengths of feats_in_asc_order and feats_in_no_order passed!")
                print("Expected: {} or {}, Received: {}".format(self.len_coeffs, self.len_coeffs - 1, len(lb)))
                return False
        elif lb != configs['low_val']:
            min_con_orig[1:] = [lb for idx, _ in enumerate(min_con_orig) if idx > 0]

        # Initialize ub constraints
        max_con_orig = [configs['high_val'] for i in range(self.len_coeffs)]
        if isinstance(ub, list):
            if len(ub) == self.len_coeffs:
                max_con_orig = ub
            elif len(ub) == self.len_coeffs - 1:
                max_con_orig[1:] = ub
            else:
                print("INCORRECT CONFIG: Number of upper bounds do not match with number of feats_in_asc_order and feats_in_no_order passed!")
                print("Expected: {} or {}, Received: {}".format(self.len_coeffs, self.len_coeffs - 1, len(ub)))
                return False
        elif ub != configs['high_val']:
            max_con_orig[1:] = [ub for idx, _ in enumerate(max_con_orig) if idx > 0]

        # Sanitize bound constraints
        for i in range(1, self.len_feats_in_asc_order + 1):
            min_con_orig[i] = max(0, min(configs['high_val'], min_con_orig[i]))
            max_con_orig[i] = max(0, min(configs['high_val'], max_con_orig[i]))

        for i in range(2, self.len_feats_in_asc_order + 1):
            min_con_orig[i] = max((1 + self.min_gap_pct[i - 1]) * min_con_orig[i - 1], min_con_orig[i])

        i = self.len_feats_in_asc_order - 1
        while i:
            max_con_orig[i] = min(max_con_orig[i + 1] / (1 + self.min_gap_pct[i]), max_con_orig[i])
            i -= 1

        if no_intercept:
            min_con_orig[0] = 0.0
            max_con_orig[0] = configs['low_val']

        # Validate bound constraints
        for i in range(self.len_coeffs):
            if max_con_orig[i] < min_con_orig[i]:
                print("INCORRECT CONFIG: Upper and lower bounds do not conform with minimum margins specified!")
                return False

        self.min_con_orig = min_con_orig
        self.max_con_orig = max_con_orig

        return True

    def _finalize_bound_constraints(self):
        self.min_con = self.min_con_orig.copy()
        self.max_con = self.max_con_orig.copy()

        if not self.anchor_coeff_idx:
            return

        for i in range(self.anchor_coeff_idx + 1, self.len_feats_in_asc_order + 1):
            self.min_con[i] = max(0,
                                  self.min_con_orig[i] - self.max_con_orig[i - 1] * (1 + self.min_gap_pct[i - 1]))
            self.max_con[i] = max(self.min_con[i] + configs['low_val'],
                                  self.max_con_orig[i] - self.min_con_orig[i - 1] * (1 + self.min_gap_pct[i - 1]))

        i = self.anchor_coeff_idx - 1
        while i:
            self.min_con[i] = max(0,
                                  self.min_con_orig[i + 1] / (1 + self.min_gap_pct[i]) - self.max_con_orig[i])
            self.max_con[i] = max(self.min_con[i] + configs['low_val'],
                                  self.max_con_orig[i + 1] / (1 + self.min_gap_pct[i]) - self.min_con_orig[i])
            i -= 1

    def _feat_eng(self, df) -> np.ndarray:
        X = df[self.feats_in_asc_order].copy()

        if self.anchor_coeff_idx:
            i = self.len_feats_in_asc_order
            while i > self.anchor_coeff_idx:
                X[f'F{i}'] = X[self.feats_in_asc_order[i - 1]]
                if i < self.len_feats_in_asc_order:
                    X[f'F{i}'] += X[f'F{i + 1}'] * (1 + self.min_gap_pct[i])
                i -= 1

            for i in range(1, self.anchor_coeff_idx):
                X[f'F{i}'] = -X[self.feats_in_asc_order[i - 1]]
                if i > 1:
                    X[f'F{i}'] += X[f'F{i - 1}'] / (1 + self.min_gap_pct[i - 1])

            X[f'F{self.anchor_coeff_idx}'] = X[self.feats_in_asc_order[self.anchor_coeff_idx - 1]]
            if self.anchor_coeff_idx < self.len_feats_in_asc_order:
                X[f'F{self.anchor_coeff_idx}'] += X[f'F{self.anchor_coeff_idx + 1}'] * (1 + self.min_gap_pct[self.anchor_coeff_idx])
            if self.anchor_coeff_idx > 1:
                X[f'F{self.anchor_coeff_idx}'] -= X[f'F{self.anchor_coeff_idx - 1}'] / (1 + self.min_gap_pct[self.anchor_coeff_idx - 1])

            X = X.drop(self.feats_in_asc_order, axis=1)

        X = pd.concat([X, df[self.feats_in_no_order].copy()], join='inner', axis=1)

        # Convert independent variables to a matrix
        X = X.values

        # Add an array of ones to act as intercept coefficient
        ones = np.ones(X.shape[0])
        # Combine array of ones and indepedent variables
        X = np.concatenate((ones[:, np.newaxis], X), axis=1)

        return X

    def _get_coeff(self, results:dict) -> bool:
        if not self.anchor_coeff_idx:
            self.coeffs = results.x.copy()
        else:
            self.coeffs[0] = results.x[0]
            self.coeffs[self.anchor_coeff_idx] = results.x[self.anchor_coeff_idx]

            for i in range(self.anchor_coeff_idx + 1, self.len_feats_in_asc_order + 1):
                self.coeffs[i] = self.coeffs[i - 1] * (1 + self.min_gap_pct[i - 1]) + results.x[i]

            i = self.anchor_coeff_idx - 1
            while i:
                self.coeffs[i] = self.coeffs[i + 1] / (1 + self.min_gap_pct[i]) - results.x[i]
                i -= 1

            for i in range(self.len_feats_in_asc_order + 1, self.len_coeffs):
                self.coeffs[i] = results.x[i]

        return self._post_hoc_corrections()

    def _post_hoc_corrections(self) -> bool:
        if self.anchor_coeff_idx:
            # Sanitize
            for i in range(1, self.len_feats_in_asc_order + 1):
                if self.coeffs[i] < self.min_con_orig[i]:
                    self.coeffs[i] = self.min_con_orig[i]
                    if i < self.len_feats_in_asc_order and self.coeffs[i + 1] < self.coeffs[i] * (1 + self.min_gap_pct[i]):
                        self.coeffs[i + 1] = self.coeffs[i] * (1 + self.min_gap_pct[i])
                elif self.coeffs[i] > self.max_con_orig[i]:
                    self.coeffs[i] = self.max_con_orig[i]
                    if i > 1 and self.coeffs[i - 1] > self.coeffs[i] / (1 + self.min_gap_pct[i - 1]):
                        self.coeffs[i - 1] = self.coeffs[i] / (1 + self.min_gap_pct[i - 1])

        # Validate
        for i in range(1, self.len_feats_in_asc_order + 1):
            if self.min_con_orig[i] - self.coeffs[i] > configs['tol'] or\
                    self.coeffs[i] - self.max_con_orig[i] > configs['tol'] or\
                    (i < self.len_feats_in_asc_order and\
                        self.coeffs[i] * (1 + self.min_gap_pct[i]) - self.coeffs[i + 1] > configs['tol']) or\
                    (i > 1 and\
                        self.coeffs[i - 1] - self.coeffs[i] / (1 + self.min_gap_pct[i - 1]) > configs['tol']):
                return False

        return True

if __name__ == '__main__':
    pass
    
