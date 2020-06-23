import numpy as np
from joblib import dump
from sklearn import svm

def SVM(X_tr, y_tr, kernel, model_name):
    y_tr = np.ravel(y_tr.to_numpy())
    
    # Train an SVM classifier with Linear kernel, gamma selected automatically
    clf = svm.SVC(kernel=kernel, class_weight='balanced')
    svm_mdl = clf.fit(X_tr, y_tr)
    dump(svm_mdl, model_name)
    return None
