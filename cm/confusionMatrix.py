from sklearn.metrics import confusion_matrix
import numpy as np
class confusionMatrix:
    def __init__(self, ytrue, ypred, labels) -> None:
        self.ytrue = ytrue
        self.ypred = ypred
        self.labels = labels
        pass
    
    def getConfusionMatrix(self):
        return confusion_matrix(self.ytrue, self.ypred, labels=self.labels)
    
    def precision(self):
        cm = self.getConfusionMatrix()
        true_pos = np.diag(cm)
        false_pos = np.sum(cm, axis=0) - true_pos
        eps = np.finfo(float).eps
        return np.sum(true_pos / (true_pos + false_pos + eps)) * 100/ 15
    
    def recall(self):
        cm = self.getConfusionMatrix()
        true_pos = np.diag(cm)
        false_neg = np.sum(cm, axis=1) - true_pos
        eps = np.finfo(float).eps
        return np.sum(true_pos / (true_pos + false_neg + eps)) * 100 / 15

    
