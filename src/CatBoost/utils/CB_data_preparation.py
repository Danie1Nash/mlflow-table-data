import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DataPrepare():
    def __init__(self, path_to_data):
        self.df = pd.read_csv(path_to_data)

    def preprocess(self):
        self.df['oper_type'] = self.df['oper_type + oper_attr'].apply(lambda x: x.split('_')[0])
        self.df['oper_attr'] = self.df['oper_type + oper_attr'].apply(lambda x: x.split('_')[1])
        self.df['oper_type'] = self.df['oper_type'].astype(int)
        self.df['oper_attr'] = self.df['oper_attr'].astype(int)
        self.df.drop('oper_type + oper_attr', axis=1, inplace=True)
        enc = LabelEncoder()
        self.df['type'] = enc.fit_transform(self.df['type'])
        self.df['priority'] = self.df['priority'].astype(int)
        self.df['is_privatecategory'].replace({'N': 1, 'Y': 2, '0': 0}, inplace=True)
        self.df['class'] = self.df['class'].astype(int)
        self.df['is_return'].replace({'N': 1, 'Y': 2}, inplace=True)
        self.df['mailtype'] = self.df['mailtype'].astype(int)
        self.df['mailctg'] = self.df['mailctg'].astype(int)
        self.df['directctg'] = self.df['directctg'].astype(int)
        self.df['postmark'] = self.df['postmark'].astype(int)
        self.df['total_qty_over_index'] = self.df['total_qty_over_index'].astype('int')
        drop_columns = ['is_in_yandex', 'dist_qty_oper_login_1', 'total_qty_oper_login_1', 'total_qty_oper_login_0',
                        'total_qty_over_index_and_type', 'is_privatecategory', 'index_oper', 'name_mfi', 'mailrank']
        self.df.drop(drop_columns, axis=1, inplace=True)
        numeric_col = ['weight', 'transport_pay', 'weight_mfi', 'price_mfi', 'total_qty_over_index']
        for col in numeric_col:
            scaler = MinMaxScaler().fit(np.array(self.df[col]).reshape(-1, 1))
            self.df[col] = scaler.transform(np.array(self.df[col]).reshape(-1, 1))
        return self.df
