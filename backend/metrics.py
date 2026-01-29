import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import io
import base64

class Metrics:
    def evaluate(self, y_test, y_pred, y_prob=None):

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }

        roc_img = None
        if y_prob is not None:
            try:
                results["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                plt.figure()
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC Curve")
                roc_img = self._save_plot()
            except:
                pass

        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_img = self._save_plot()

        return results, roc_img, cm_img

    def _save_plot(self):
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
