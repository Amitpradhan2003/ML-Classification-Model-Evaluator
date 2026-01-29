from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC

class ModelDevelopment:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def get_model(self):
        models = {
            "decision_tree": DecisionTreeClassifier(),
            "naive_bayes": GaussianNB(),
            "knn": KNeighborsClassifier(),
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(),
            "bagging": BaggingClassifier(),
            "svm": SVC(probability=True)
        }
        self.model = models[self.model_name]
        return self.model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print(f"✅ {self.model_name} model trained")
        return self.model
