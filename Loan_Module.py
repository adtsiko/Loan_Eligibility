import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math


class LoanScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.columns = columns
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.var_ = np.var(X[self.columns])
        self.mean_ = np.mean(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        X = X.reset_index().drop(['index'], axis=1)
        init_col_order = X.columns
        X_not_scaled = X[[_ for _ in X.columns.values if _ not in self.columns]]
        scaled_inputs = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        return pd.concat([X_not_scaled, scaled_inputs], axis=1)[init_col_order]

    def create_model(self, data):
        data = data.sample(frac=1, random_state=42)
        # targets
        targets = data['BAD']
        # inputs
        unscaled_inputs = data.iloc[:, 1:]
        loan_scaler = LoanScaler(self.columns)
        loan_scaler.fit(unscaled_inputs)
        scaled_inputs = loan_scaler.transform(unscaled_inputs)
        x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=42)
        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        reg.score(x_train, y_train)
        summary_table = pd.DataFrame(columns=['Feature name'], data=scaled_inputs.columns)
        summary_table['Coefficient'] = np.transpose(reg.coef_)
        summary_table.index = summary_table.index + 1
        summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
        summary_table = summary_table.sort_index()
        summary_table['Odd Ratio'] = np.exp(summary_table['Coefficient'])
        summary_table = summary_table.sort_values(by='Odd Ratio', ascending=False)
        print(summary_table)
        print('\nLogisticRegression model accuracy: {:.2f}%'.format(reg.score(x_test, y_test) * 100))
        with open('model', 'wb') as model_file, open('Loan_scaler', 'wb') as scaler_file:
            pickle.dump(reg, model_file)
            pickle.dump(loan_scaler, scaler_file)


class loan_model:

    def __init__(self, model_file, scaler_file, data):
        with open('model', 'rb') as model_file, open('Loan_scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scale = pickle.load(scaler_file)
            self.data = data
            self.input_data = self.scale.fit(self.data)

    def predicted_probability(self):
        if self.data is not None:
            pred = self.reg.predict_proba(self.input_data)[:, 1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.input_data)
            return pred_outputs

    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if self.data is not None:
            prob_data = self.reg.predict_proba(self.input_data)[:, 1]
            predict_data = self.reg.predict(self.input_data)
            return pd.concat([predict_data, prob_data], columns=['Prediction', 'Probability'])


class new_models:

    def __init__(self, filename, method):
        self.data = pd.read_csv(filename)
        self.method = method

    def columns_return(func):
        def wrapper(self):
            unscaled_inputs = self.data.iloc[:, 1:]
            if self.method > 0:
                self.majority = self.data[self.data['BAD'] == 0]
                self.minority = self.data[self.data['BAD'] == 1]
            columns_to_omit = ['Reason_1', 'Reason_2', 'JOB']
            self.columns = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
            result = (func(self), self.columns)
            return result

        return wrapper

    @columns_return
    def oversample(self):
        # Upsample minority class
        minority_upsampled = resample(self.minority,
                                      replace=True,  # sample with replacement
                                      n_samples=self.majority['BAD'].size,  # to match majority class
                                      random_state=42)  # reproducible results

        # Combine majority class with upsampled minority class
        self.data = pd.concat([self.majority, minority_upsampled])
        return self.data

    @columns_return
    def undersample(self):
        # Upsample minority class
        majority_upsampled = resample(self.majority,
                                      replace=True,  # sample with replacement
                                      n_samples=self.minority['BAD'].size,  # to match majority class
                                      random_state=42)  # reproducible results

        # Combine majority class with upsampled minority class
        self.data = pd.concat([self.minority, majority_upsampled])
        return self.data, self.columns

    @columns_return
    def change_perfomance_metrics(self):
        return self.data

    def clean_process_data(self):
        df = self.data
        # There a many missing data in the dataset espeacially in the 'DEBTINC' column
        # The 'Reason' and 'Job' values is better represented numerically
        # The 'BAD' column is the target
        # 'DEBTINC' column dropped
        df = df.drop(['DEBTINC'], axis=1)
        # df['REASON'].unique(), ['HomeImp', nan, 'DebtCon'], nan will be taken as other
        # 'HomeImp'=1, 'DebtCon'=0, 'nan'=2
        ''' As dummy varibles are to be created for the 'REASON' column 'DebtCon' is set to 0 as it will be the main bias of                     comparison 
            as it will be dropped to stop multicollinearity '''
        df['REASON'] = df['REASON'].map({'HomeImp': 1, 'DebtCon': 0, np.nan: 2})
        df['REASON'].unique()  # [0, 1, 2]
        # drop all rows that have an nan value
        df = df.dropna()
        # df['JOB'].unique(),  ['Other', 'Office', 'Mgr', 'ProfExe', 'Self', 'Sales']
        df['JOB'] = df['JOB'].map({'Other': 0, 'Office': 1, 'Mgr': 2, 'ProfExe': 3, 'Self': 4, 'Sales': 5})
        # 'Reason' for loan dummies
        reason_dummies = pd.get_dummies(df['REASON'], drop_first=True)
        df = pd.concat([df, reason_dummies], axis=1)
        df.columns = ['BAD', 'LOAN', 'MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG',
                      'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'Reason_1', 'Reason_2']
        # The 'REASON' column is dropped for there is now no need for it
        df = df.drop(['REASON'], axis=1)
        # The 'REASON 1 & 2' columns are reordered to were the 'REASON' columns was at
        df_columns_reordered = ['BAD', 'LOAN', 'MORTDUE', 'VALUE', 'Reason_1', 'Reason_2', 'JOB', 'YOJ', 'DEROG',
                                'DELINQ', 'CLAGE', 'NINQ', 'CLNO']
        df = df[df_columns_reordered]
        self.data = df
        if self.method < 1:
            return df
        # Methods
