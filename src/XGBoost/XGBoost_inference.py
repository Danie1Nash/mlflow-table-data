import os
from dotenv import load_dotenv

import mlflow
import yaml

load_dotenv('./.env')


class XGBoostCLF:

    def __init__(self):
        with open('../src/configs/model.yaml') as file:
            self.params_model = yaml.safe_load(file)['XGBoost']

        remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment("pochta-task")

        self.model = mlflow.xgboost.load_model(self.params_model['MODELS_PATH'])

    def predict(self, data, threshold: bool = False):
        if threshold:
            return self.model.predict(data)
        else:
            proba = self.model.predict_proba(data)
            predict_threshold = []
            for value in proba[:, 1]:
                if value > self.params_model['threshold']:
                    predict_threshold.append(1)
                else:
                    predict_threshold.append(0)

            return predict_threshold
