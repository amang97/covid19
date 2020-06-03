from .utilities.config import FP, DP
from .utilities.data_loader import DataLoader

def main():
    # Load the dataset
    dl = DataLoader()
    dl.read_data(FP['DATA'], fl=DP['FL'], ll=DP['LL'])

    # stratified splitting of the dataset
    X_tr, X_t, y_tr, y_t = dl.stratified_split(DP['SR'], DP['SEED'])

    return 0

if __name__ == "__main__":
    main()