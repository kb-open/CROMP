# Implementation of CROMP prediction
# Copyright (c) Kaushik Bar
# Licensed under the MIT license
# Author: Kaushik Bar (email: kb.opendev@gmail.com)

from joblib import load
import pandas as pd

class CROMPPredict():
    def __init__(self, model:object, verbose:bool=False):
        try:
            if isinstance(model, str):
                self.cromp_model = load(model.rstrip('/cromp_model.joblib').rstrip('/') + '/cromp_model.joblib')
            else:
                self.cromp_model = model
        except:
            print("ERROR: Model could not be loaded!")
        self.verbose = verbose

    def predict(self, df:pd.DataFrame) -> pd.Series:
        coeffs = self.cromp_model['coeffs']
        feats = self.cromp_model['feats']

        result = df.apply(lambda row: coeffs[0] + sum([x * y for x, y in zip(coeffs[1:], row[feats])]),
                          axis=1)
        if self.verbose:
            print("\n\nPredicted result:\n", result)

        return result

if __name__ == '__main__':
    pass
    
