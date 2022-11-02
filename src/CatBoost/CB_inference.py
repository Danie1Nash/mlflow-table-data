import mlflow


class CatBoost():
    def __init__(self,
                 logged_model: str = "runs:/781904df5e8d43989ce0909f75b7875b/log/catbost-11-02-2022-18-49-28/models"):
        self.logged_model = logged_model
        self.model = mlflow.catboost.load_model(self.logged_model)

    def predict(self, data):
        return self.model.predict(data)
