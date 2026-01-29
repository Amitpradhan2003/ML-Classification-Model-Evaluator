class Predictions:
    def __init__(self, model):
        self.model = model

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        return None
