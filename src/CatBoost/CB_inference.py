import mlflow
import yaml


class CatBoostCLF:
    def __init__(self):
        with open('./src/configs/model.yaml') as file:
            params_model = yaml.safe_load(file)['Catboost']

        self.model = mlflow.catboost.load_model(params_model['logged_model'])

    def predict(self, data):
        return self.model.predict(data)
