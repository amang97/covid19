import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--testSVM',
                        action='store_true',
                        help='tests saved model')

    parser.add_argument('--testRF',
                        action='store_true',
                        help='tests saved rf model')


    parser.add_argument('--dataAnalytics',
                        action='store_true',
                        help='Prints SNR, ANOVA analysis, selected features')

    parser.add_argument('--trainNN',
                        action='store_true',
                        help='Trains a feed forward Neural NeT')

    parser.add_argument('--trainRF',
                        action='store_true',
                        help='Trains a random forest classifier with 100 nodes')

    parser.add_argument('--trainSVM',
                action='store_true',
            help='trains the SVM model')

    return parser.parse_args()