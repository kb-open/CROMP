# Implementation of CROMP prediction
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

import pandas as pd

class CROMPPredict():
    def __init__(self, coeffs:[float], verbose:bool=False):
        self.verbose = verbose
        self.coeffs = coeffs

    def predict(self, df:pd.DataFrame, feature_cols_in_asc_order:[str]) -> (bool, pd.Series):
        ret_success = True
        features = feature_cols_in_asc_order

        # Initialize result with the intercept only
        result = pd.Series([self.coeffs[0] for i in range(len(df))])

        if len(self.coeffs) != len(features) + 1:
            print("INCORRECT PARAMS: Number of features do not match with number of coefficients passed!")
            ret_success = False
        else:
            result = df.apply(lambda row: self.coeffs[0] + sum([x * y for x, y in zip(self.coeffs[1:], row[features])]),
                              axis=1)
            if self.verbose:
                print("\n\nPredicted result:\n", result)

        return ret_success, result

if __name__ == '__main__':
    pass
    
