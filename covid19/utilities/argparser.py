import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--trainSVM',
            action='store_true',
            help='trains the SVM model')
        
    return parser.parse_args()