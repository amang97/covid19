# File paths
FP = {}
FP['DATA'] = './../../../Desktop/mngo/research/covid19/COVID_machine_learning_project/Data_and_data_dictionary/5.29.2020/COVID19RiskFactors-raw data_2020.5.29.csv'

# Saved Models
FP['SVM_RBF'] = './saved_models/SVM/rbf.joblib'
FP['SVM_LINEAR'] = './saved_models/SVM/linear.joblib'
FP['RFMDL'] = './saved_models/RF/rf.joblib'
FP['RFIMG'] = './saved_models/RF/anova.png'
FP['CORR'] = './results/correlation/corr.csv'
FP['COR_HMP'] = './results/correlation/cor_hmp.png'
FP['SNR'] = './results/snr/snr_fs.png'
FP['UBE'] = './results/ube/ube_fs.png'

# Data Parameters
DP = {}
DP['ROUND'] = 3
DP['FL'] = ['adm_insulin', 'bmi', 'adm_nausea', 'arrival_o2therapy', 'sex',\
            'arrival_bps', 'adm_diarrhea', 'arrival_o2']
DP['LL'] = ['icu_admission'] # ['intubation_status']
DP['NUM_CON'] = 3
DP['CON_FS_MODE'] = 'snr'
DP['CONFL'] = ['age', 'arrival_temp', 'arrival_hr', 'arrival_rr', 'arrival_bps',\
            'arrival_bpd', 'arrival_o2','height', 'weight', 'bmi']
DP['NUM_CAT'] = 5
DP['CAT_FS_MODE'] = 'ube'
DP['CATFL'] = [\
            'race',\
            'ethnicgroup',\
            'sex',\
            'arrival_o2therapy',\
            'bmi_category',\
            'symptom_duration',\
            'adm_fever', 'adm_cough', 'adm_sputum', 'adm_dyspnea', 'adm_uri',\
                'adm_chest_pain', 'adm_fatigue', 'adm_myalgia', 'adm_anorexia',\
                'adm_abd', 'adm_diarrhea', 'adm_vomit', 'adm_nausea',\
                'adm_anosmia', 'adm_chills', 'adm_photo', 'adm_phono',\
                'adm_neck', 'adm_confusion', 'adm_headache',\
            'smoking_status',\
            'hcw',\
            'exposure',\
            'homeless',\
            'adm_aspirin', 'adm_nsaid', 'adm_ace', 'adm_arb', 'adm_betablock',\
                'adm_calcium', 'adm_diuretic', 'adm_antihypertensive', 'adm_statin',\
                'adm_oral_hypoglycemic', 'adm_insulin', 'adm_oral_corticosteroid',\
                'adm_inhaled_corticosteroid', 'adm_hydroxyurea', 'adm_colchicine',\
                'adm_hydroxychloroquine', 'adm_art', 'adm_azithro', 'adm_sarilumab',\
                'adm_tocilizumab', 'adm_anakinra', 'adm_siltuximab',\
            'rf_htn', 'rf_cardiac', 'rf_echo', 'rf_osa', 'rf_asthma', 'rf_pulmonary',\
            'rf_diabetes', 'rf_ckd', 'rf_dialysis', 'rf_cirrhosis', 'rf_dementia',\
            'rf_neuro___0', 'rf_neuro___1', 'rf_neuro___2', 'rf_neuro___3',\
            'rf_vascular___0', 'rf_vascular___1', 'rf_vascular___2',\
            'rf_sicklecell', 'rf_hiv', 'rf_osteoarthritis',\
            'rf_autoimmune___0', 'rf_autoimmune___1', 'rf_autoimmune___2',\
                'rf_autoimmune___3', 'rf_autoimmune___4', 'rf_autoimmune___5',\
                'rf_autoimmune___6',\
            'rf_autoinflammatory___0', 'rf_autoinflammatory___1',\
                'rf_autoinflammatory___2', 'rf_autoinflammatory___3',\
                'rf_autoinflammatory___4', 'rf_autoinflammatory___5',\
            'rf_malignancy',\
            'rf_transplant',\
            'rf_alc',\
            'comp_resp_panel',\
            'crp_result___0', 'crp_result___1', 'crp_result___2',\
                'crp_result___3', 'crp_result___4', 'crp_result___5',\
                'crp_result___6', 'crp_result___7', 'crp_result___8',\
                'crp_result___9', 'crp_result___10',\
            'other_infectious___0', 'other_infectious___1',\
                'other_infectious___2', 'other_infectious___3',\
                'other_infectious___4', 'other_infectious___5']
DP['SR'] = 0.20
DP['SEED'] = 1234