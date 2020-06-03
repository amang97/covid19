from joblib import load
from sklearn.metrics import accuracy_score

from .ml.svm import SVM
from .utilities.config import FP, DP
from .utilities.data_loader import DataLoader

def main():
    # Load the dataset
    dl = DataLoader()
    dl.read_data(FP['DATA'], fl=DP['FL'], ll=DP['LL'])

    # Stratified Splitting of the dataset
    X_tr, X_t, y_tr, y_t = dl.stratified_split(DP['SR'], DP['SEED'])

    # Pre-Processing

    # Train SVM classifiers
    SVM(X_tr, y_tr, X_t)
    
    # Load Saved SVM models and measure accuracies
    rbf_svm_mdl = load(FP['SVM_RBF'])
    y_p = rbf_svm_mdl.predict(X_t)
    print(f'RBF Kernel SVM Accuracy: {accuracy_score(y_p, y_t)}')
    
    linear_svm_mdl = load(FP['SVM_LINEAR'])
    y_p = linear_svm_mdl.predict(X_t)
    print(f'RBF Linear SVM Accuracy: {accuracy_score(y_p, y_t)}')
    return 0

if __name__ == "__main__":
    main()