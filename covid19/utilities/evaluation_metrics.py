import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from .config import FP

def tn(y_pred, y_true): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_pred, y_true): return confusion_matrix(y_true, y_pred)[0, 1]
def specificity(y_pred, y_true): return tn(y_true, y_pred)/ (tn(y_true, y_pred) + fp(y_true, y_pred))

class EvaluationMetrics:
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.metrics['accuracy'] = []
        self.metrics['precision'] = []
        self.metrics['sensitivity'] = []
        self.metrics['specificity'] = []
        self.metrics['f1'] = []
    
    def evaluate_all_metrics(self, y_p, y_t, r):
        self.metrics['accuracy'].append(round(accuracy_score(y_p, y_t),r))
        self.metrics['precision'].append(round(precision_score(y_p, y_t),r))
        self.metrics['sensitivity'].append(round(recall_score(y_p, y_t),r))
        self.metrics['specificity'].append(round(specificity(y_p, y_t),r))
        self.metrics['f1'].append(round(f1_score(y_p, y_t),r))

    def accuracy(self):
        return self.metrics['accuracy']
    def precision(self):
        return self.metrics['precision']
    def sensitivity(self):
        return self.metrics['sensitivity']
    def specificity(self):
        return self.metrics['specificity']
    def f1(self):
        return self.metrics['f1']
    
    def __str__(self):
        s = '\n'+ f'Accuracy: {self.accuracy()}'+'\n\n'\
            + f'Precision: {self.precision()}'+'\n\n'\
            + f'Sensitivity/Recall: {self.sensitivity()}'+'\n\n'\
            + f'Specificity/Selectivity: {self.specificity()}'+'\n\n'\
            + f'F1: {self.f1()}' + '\n\n'
        return s
    
    def max_accuracy(self):
        return max(self.accuracy()), self.accuracy().index(max(self.accuracy()))
    
    def max_precision(self):
        return max(self.precision()), self.precision().index(max(self.precision()))
    
    def max_sensitivity(self):
        return max(self.sensitivity()), self.sensitivity().index(max(self.sensitivity()))
    
    def max_specificity(self):
        return max(self.specificity()), self.specificity().index(max(self.specificity()))
    
    def max_f1(self):
        return max(self.f1()), self.f1().index(max(self.f1()))
    
    def max_metrics(self):
        acc, acci = self.max_accuracy()
        pre, prei = self.max_precision()
        sen, seni = self.max_sensitivity()
        spe, spei = self.max_specificity()
        f1, f1i = self.max_f1()
        s = '\n' + f'Maximum Accuracy: {acc} at combination {acci}' + '\n'\
            + f'Maximum precision: {pre} at combination {prei}' + '\n'\
            + f'Maximum sensitivity: {sen} at combination {seni}' + '\n'\
            + f'Maximum specificity: {spe} at combination {spei}' + '\n'\
            + f'Maximum f1: {f1} at combination {f1i}' + '\n'\
            + '\n---------------**********---------------\n'
        return s
    
    def visualize(self, model_name):
        fig = plt.figure()
        plt.plot(range(len(self.accuracy())), self.accuracy(), label='accuracy')
        plt.plot(range(len(self.precision())), self.precision(), label='precision')
        plt.plot(range(len(self.sensitivity())), self.sensitivity(), label='sensitivity')
        plt.plot(range(len(self.specificity())), self.specificity(), label='specificity')
        plt.plot(range(len(self.f1())), self.f1(), label='f1')
        plt.legend()
        plt.savefig(FP['METRICS'] + model_name + '.png')
