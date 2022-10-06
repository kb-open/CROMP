[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9-blue.svg)](https://pypi.org/project/cromp/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/kb-open/CROMP/blob/main/LICENSE.txt)
[![PyPI - PyPi](https://img.shields.io/pypi/v/cromp)](https://pypi.org/project/cromp/)
[![Build](https://img.shields.io/github/workflow/status/kb-open/CROMP/Test%20Build/main)](https://github.com/kb-open/CROMP/actions/)

# ✏️ CROMP (Constrained Regression with Ordered and Margin-sensitive Parameters) 
This repo provides the official implementation of CROMP (Constrained Regression with Ordered and Margin-sensitive Parameters) along with the test dataset and the experimental test pipeline code to reproduce the benchmarking results as described in the CROMP paper ("**Constrained Regression with Ordered and Margin-sensitive Parameters: Application in improving interpretability for wage models with prior knowledge**").

Author: [Kaushik Bar](kb.opendev@gmail.com)

---------------------------
## 🎯 Quick Info
CROMP allows: 
* User-defined order among the coefficients
* User-defined minimum margins (i.e., percentage gaps) between the coefficients
* User-defined lower and upper bounds for each coefficient

This package is both interpretable and accurate.
There is no other package / library available in the public domain today with these capabilities.

---------------------------
## 💻 Installation
Install using [PyPI](https://pypi.org/project/cromp/):
```bash
pip install cromp
```

---------------------------
## 💻 Dependencies
The package depends on:
* python >= 3.6
* scipy
* numpy
* pandas

The tests depend additionally on:
* scikit-learn
* [Data scientist salary data](https://www.kaggle.com/datasets/nikhilbhathi/data-scientist-salary-us-glassdoor?select=data_cleaned_2021.csv)
* [Ames House price data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv)

---------------------------
## 🚀 Usage
Below is a minimal usage guide. This library aloows flexibility for more fine-grained controls for each coefficient separately.
Please go through the [tests](https://github.com/kb-open/cromp/blob/main/tests/cromp_tests.py) in addition to see more examples.

```python
>>> from cromp import CROMPTrain, CROMPPredict

>>> df = pd.read_csv("tests/data/ames_house_prices_data.csv")
>>> df_train = df.iloc[:-50, :]
>>> df_test = df.iloc[-50:, :]

>>> target_col = 'SalePrice'
>>> feature_cols_in_asc_order = ['1stFlrSF', 'TotalBsmtSF', 'GrLivArea']

>>> # Instantiate and configure
>>> model = CROMPTrain()
>>> ret_success = model.configure(df_train, target_col, feature_cols_in_asc_order, min_gap_pct=0.5, lb=0.0, ub=100.0, no_intercept=True)

>>> # Train
>>> ret_success, wages = model.train()

>>> # Predict
>>> model = CROMPPredict(wages)
>>> ret_success, result = model.predict(df_test, feature_cols_in_asc_order)
```

---------------------------
## ☕ Contact
Feel free to contact [Kaushik Bar](https://www.linkedin.com/in/iamkbar/) if you have any further questions / feedback.

