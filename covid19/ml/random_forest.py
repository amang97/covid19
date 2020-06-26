import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from ..utilities.config import DP
def random_forest(X_tr, y_tr, model_name):
    y_tr = np.ravel(y_tr.to_numpy())
    rfc = RandomForestClassifier(random_state=DP['SEED'],\
                                    class_weight='balanced')
    rfc_mdl = rfc.fit(X_tr, y_tr)
    dump(rfc_mdl, model_name)
    return None