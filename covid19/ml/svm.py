import numpy as np
from joblib import dump
from sklearn import svm

from ..utilities.config import FP

def SVM(X_tr, y_tr):
    y_tr = np.ravel(y_tr.to_numpy())

    # Train an SVM classifier with RBF kernel, gamma selected automatically
    rbf_clf = svm.SVC(kernel='rbf')
    rbf_svm_mdl = rbf_clf.fit(X_tr, y_tr)
    dump(rbf_svm_mdl, FP['SVM_RBF'])

    # Train an SVM classifier with Linear kernel, gamma selected automatically
    linear_clf = svm.SVC(kernel='linear')
    linear_svm_mdl = linear_clf.fit(X_tr, y_tr)
    dump(linear_svm_mdl, FP['SVM_LINEAR'])
    return None