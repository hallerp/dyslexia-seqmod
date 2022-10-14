import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import numpy as np
from datetime import datetime


class ROC:
    def __init__(
        self,
        model,
        wordvectors,
        tune,
        mean_fpr=np.linspace(0, 1, 100),
        tprs: list = [],
        aucs: list = [],
        mean_auc: int = 0,
        std_auc: int = 0,
    ):
        self.tune = 'tuned' if tune else 'untuned'
        self.model = model
        self.wordvectors = wordvectors
        self.title = f"ROC {self.model} {self.wordvectors} {self.tune}"
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.tprs = tprs
        self.aucs = aucs
        self.mean_fpr = mean_fpr
        self.mean_auc = mean_auc
        self.std_auc = std_auc

    def update(self, tpr, auc):
        self.tprs.append(tpr)
        self.aucs.append(auc)

    def plot(self):
        self.ax.plot(
            [0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8
        )
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        self.mean_auc = auc(self.mean_fpr, mean_tpr)
        self.std_auc = np.std(self.aucs)
        self.ax.plot(
            self.mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (self.mean_auc, self.std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        self.ax.fill_between(
            self.mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        self.ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=self.title,
        )
        self.ax.legend(loc="lower right")

    def show(self):
        self.fig.show()

    def save(self):
        date = datetime.now().strftime("%Y%m%d-%H:%M")
        plt.savefig(f"roc_{date}_{self.model}_{self.wordvectors}_{self.tune}.png", bbox_inches="tight", dpi=500)

    def get_tprs_aucs(self, y_trues, y_preds, test_fold):
        fpr, tpr, _ = roc_curve(y_trues, y_preds, drop_intermediate=False)
        viz = RocCurveDisplay.from_predictions(
            y_trues,
            y_preds,
            name="ROC fold {}".format(test_fold + 1),
            alpha=0.3,
            lw=1,
            ax=self.ax,
        )
        interp_tpr = np.interp(self.mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        self.update(interp_tpr, viz.roc_auc)
