# File paths
FP = {}
FP['DATA'] = './../../../Desktop/mngo/research/covid19/COVID_machine_learning_project/Data_and_data_dictionary/5.29.2020/COVID19RiskFactors-raw data_2020.5.29.csv'

# Data set Parameters
DP = {}
DP['FL'] = ['adm_insulin', 'bmi', 'adm_nausea', 'arrival_o2therapy', 'sex',\
            'arrival_bps', 'adm_diarrhea', 'los', 'arrival_o2']
DP['LL'] = ['icu_admission'] # 'intubation_status'
DP['SR'] = 0.20
DP['SEED'] = 1234