import pandas as pd
import yaml


class DataPrepare:

    def __init__(self, path_to_data):
        self.df = pd.read_csv(path_to_data, low_memory=False)

        with open("../src/configs/data_info.yaml", 'r') as file:
            self.data_need = yaml.safe_load(file)

        for col in self.df:
            if col not in list(self.data_need['numeric'].keys()) + list(self.data_need['category'].keys()):
                if col != 'oper_type + oper_attr':
                    self.df.drop(col, axis=1, inplace=True)

    def _min_max_scaler(self, col: str, max_value: float, min_value: float) -> None:
        self.df[col] = self.df[col].apply(lambda value: (value - min_value) / (max_value - min_value))

    def preprocess(self):
        # numeric field
        for num_col in self.data_need['numeric']:
            self._min_max_scaler(
                num_col, self.data_need['numeric'][num_col]['max'], self.data_need['numeric'][num_col]['min']
            )

        # category field
        self.df['type'] = self.df['type'].replace(self.data_need['category']['type'])
        self.df['priority'] = self.df['priority'].astype(int)
        self.df['class'] = self.df['class'].astype(int)
        self.df['is_return'] = self.df['is_return'].replace(self.data_need['category']['is_return'])
        self.df['mailtype'] = self.df['mailtype'].astype(int)
        self.df['mailctg'] = self.df['mailctg'].astype(int)
        self.df['directctg'] = self.df['directctg'].astype(int)
        self.df['postmark'] = self.df['postmark'].astype(int)
        self.df['oper_type'] = self.df['oper_type + oper_attr'].apply(lambda x: x.split('_')[0])
        self.df['oper_attr'] = self.df['oper_type + oper_attr'].apply(lambda x: x.split('_')[1])
        self.df['oper_type'] = self.df['oper_type'].astype(int)
        self.df['oper_attr'] = self.df['oper_attr'].astype(int)
        self.df.drop('oper_type + oper_attr', axis=1, inplace=True)

        return self.df
