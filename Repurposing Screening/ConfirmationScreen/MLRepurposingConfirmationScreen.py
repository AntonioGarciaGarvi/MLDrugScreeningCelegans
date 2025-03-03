import os
import numpy as np
import pandas as pd
from tierpsytools.preprocessing.scaling_class import scalingClass
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
import random
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from collections import Counter
from tierpsytools.preprocessing.filter_data import drop_ventrally_signed, filter_nan_inf, cap_feat_values, feat_filter_std

if __name__ == "__main__":
    ML_algorithm = 'RF'
    # ==============================================================================
    # Data Load and cleaning
    # ==============================================================================
    # Seed for reproducible results
    seed = 20
    np.random.seed(seed)
    random.seed(seed)

    # Drugs dataframe
    drugs_codes = {
        "MRT ID": [
            "MRT00012372", "MRT00013320", "MRT00034657", "MRT00166891", "MRT00208613",
            "MRT00012996", "MRT00208714", "MRT00012420", "MRT00012549", "MRT00170216",
            "MRT00170379", "MRT00208875", "MRT00208704", "MRT00208814", "MRT00208833",
            "MRT00170380", "MRT00022743", "MRT00208555", "MRT00170198", "MRT00208695",
            "MRT00012785", "MRT00208643", "MRT00012550", "MRT00170362", "MRT00170492",
            "MRT00013208", "MRT00013702", "MRT00170396", "MRT00033721", "MRT00033796",
            "MRT00033979", "MRT00034183", "MRT00034607", "MRT00034695", "MRT00170329",
            "MRT00208812", "MRT00208813", "MRT00208822", "MRT00208842"
        ],
        "Drug": [
            "Abitrexate", "Sulindac", "Rofecoxib", "Mesalamine", "Mitotane",
            "Loratadine", "Azatadine dimaleate", "Amitriptyline HCl", "Clozapine", "Olanzapine",
            "Rizatriptan benzoate", "Vinblastine", "Detomidine HCl", "Medetomidine HCl", "Ivabradine HCl",
            "Ziprasidone hydrochloride", "Azacyclonol", "Iloperidone", "Mirtazapine", "Atorvastatin calcium",
            "Fenofibrate", "Liranaftate", "D-Cycloserine", "Sulfadoxine", "Daunorubicin HCl",
            "Ofloxacin", "Idarubicin", "Moxifloxacin", "Carbenicillin disodium", "Sarafloxacin HCl",
            "Roxithromycin", "Norfloxacin", "Ciprofloxacin", "Pefloxacin mesylate", "Fleroxacin",
            "Besifloxacin HCl", "Enrofloxacin", "Nadifloxacin", "Sitafloxacin hydrate"
        ]
    }

    drugs_codes = pd.DataFrame(drugs_codes)

    ###### CONFIRMATION SCREEN PATHS #####
    REPURPOSING_DATASET_ROOT_DIR = '/DataSets/DrugRepurposing/ConfirmationScreen/'
    METADATA_PATH = REPURPOSING_DATASET_ROOT_DIR + 'metadata.csv'
    FEATS_PATH = REPURPOSING_DATASET_ROOT_DIR  +  'features.csv'
    FNAMES_PATH = REPURPOSING_DATASET_ROOT_DIR + 'filenames.csv'
    SCREEN = 'ConfirmationScreen'


    # Path to save results
    res_dir = '/Repurposing_results/' + SCREEN + '/' +  ML_algorithm + '/'

    # Read hydra metadata
    repurposing_feat_df, repurposing_metadata = read_hydra_metadata(FEATS_PATH, FNAMES_PATH, METADATA_PATH)

    # Change names
    repurposing_metadata["drug_type"] = repurposing_metadata["drug_type"].map(drugs_codes.set_index("MRT ID")["Drug"]).fillna(repurposing_metadata["drug_type"])

    light_type = 'bluelight'
    # light_type = 'align'

    if light_type in ['prestim', 'bluelight', 'poststim']:    # Get only bluelight features
        repurposing_metadata = repurposing_metadata[repurposing_metadata['bluelight'] == light_type]
        repurposing_feat_df = repurposing_feat_df.loc[repurposing_metadata.index]
    else:
        # Concatenates the features from the three bluelight conditions for each well
        repurposing_feat_df, repurposing_metadata = align_bluelight_conditions(repurposing_feat_df, repurposing_metadata, how='inner')

    #  Count the number of wells per compound
    counts_by_compound = repurposing_metadata['drug_type'].value_counts()
    counts_by_compound.to_excel('counts_by_compound.xlsx')

    # Drop compounds not in FDA approved library- from manual inspection these
    # did not rescue the phenotype we are looking for
    mask = repurposing_metadata['drug_type'].isin(['Procaine_HCl',
                                   'Ambroxol_HCl',
                                   'Carbamazepine'])

    repurposing_feat_df = repurposing_feat_df[~mask]
    repurposing_metadata = repurposing_metadata[~mask]

    # Drop wells containing neither DMSO or a compound from both dataframes
    mask = repurposing_metadata['drug_type'].isin(['empty'])
    repurposing_feat_df = repurposing_feat_df[~mask]
    repurposing_metadata = repurposing_metadata[~mask]

    # Filter wells 
    mask = repurposing_metadata['well_label'].isin([1.0, 3.0])

    repurposing_feat_df = repurposing_feat_df[mask]
    repurposing_metadata = repurposing_metadata[mask]

    # Filter parameters

    min_nan_value = 0.3 # ratio of features that could not be evaluated
    max_nan_ratio = 0.05

    # remove features and wells with too many nans and std=0
    # Filter nan inf data  0 --> filter features, 1 --> filter samples
    repurposing_feat_df = filter_nan_inf(repurposing_feat_df, min_nan_value, axis=1)
    repurposing_feat_df = filter_nan_inf(repurposing_feat_df, max_nan_ratio, axis=0)

    repurposing_metadata = repurposing_metadata.loc[repurposing_feat_df.index]

    repurposing_feat_df = feat_filter_std(repurposing_feat_df)
    # replace features with too large values (>cutoff) with the max value of the given feature in the remaining data points.
    repurposing_feat_df = cap_feat_values(repurposing_feat_df)
    repurposing_feat_df = drop_ventrally_signed(repurposing_feat_df)

    repurposing_metadata = repurposing_metadata.loc[repurposing_feat_df.index]

    # abs features no longer in tierpsy
    pathcurvature_feats = [x for x in repurposing_feat_df.columns if 'path_curvature' in x]
    # remove these features
    repurposing_feat_df = repurposing_feat_df.drop(columns=pathcurvature_feats)

    counts_by_compound = repurposing_metadata['drug_type'].value_counts()

    # Add a column with the classes we define
    repurposing_metadata.loc[:, 'Class'] = repurposing_metadata['worm_gene'].apply(lambda x: 0 if x == 'N2' else 1)
    Yrepurposing = repurposing_metadata.Class

    # Metadata to train the model (controls N2 and unc-80 in DMSO)
    metadata_model_fit = repurposing_metadata[repurposing_metadata['drug_type'].isin(['DMSO'])]
    feat_df_mfit = repurposing_feat_df.loc[metadata_model_fit.index]
    Y = metadata_model_fit.Class

    # Balance classes
    print("Dataset set class distribution before balancing: ", Counter(Y))

    # Initialize the under-sampler with a fixed random seed for reproducibility
    undersampler = RandomUnderSampler(random_state=seed)

    feat_df_mfit, Y = undersampler.fit_resample(feat_df_mfit, Y)
    metadata_model_fit = metadata_model_fit.loc[feat_df_mfit.index]

    # feat_df_train, feat_df_val, Ytrain, Ytest = train_test_split(
    #     feat_df_mfit, Y, test_size=0.2, random_state=seed, stratify=Y)
    # Lo separo porque la función train_test_split no garantiza el balanceo exacto por clases,
    #  hace una aproximación con el parámetro stratify, pero no lo hace exacto

    # Separate by class
    class0 = feat_df_mfit[Y == 0]
    class1 = feat_df_mfit[Y == 1]

    # Split each class individually
    class0_train, class0_test = train_test_split(class0, test_size=0.2, random_state=seed)
    class1_train, class1_test = train_test_split(class1, test_size=0.2, random_state=seed)

    # Combine the splits
    feat_df_train = pd.concat([class0_train, class1_train])
    feat_df_val = pd.concat([class0_test, class1_test])

    metadata_train = metadata_model_fit.loc[feat_df_train.index]
    metadata_test = metadata_model_fit.loc[feat_df_val.index]

    # Impute NaNs and Inf
    feat_df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df_val.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fit imputer only on training data
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(feat_df_train)  # Compute means from training data

    # Apply the same imputer to both train and test data
    feat_df_train = pd.DataFrame(imputer.transform(feat_df_train), columns=feat_df_train.columns)
    feat_df_val = pd.DataFrame(imputer.transform(feat_df_val), columns=feat_df_val.columns)

    Ytrain = metadata_train.Class
    Ytest = metadata_test.Class

    print("Dataset set class distribution after balancing: ", Counter(Y))
    print("Train set class distribution: ", Counter(Ytrain))
    print("Test set class distribution: ", Counter(Ytest))


    # ==============================================================================
    # Feature selection
    # ==============================================================================

    estimator = RandomForestClassifier(random_state=seed)

    n_feat_to_select = [feat_df_train.shape[1]] + [2**i for i in range(11, 6, -1)]
    cv_scores = []
    feat_set = [feat_df_train.columns.to_list()]
    scaler = scalingClass(scaling='standardize')  


    pipe = Pipeline([
        ('scaler', scaler),
        ('estimator', estimator)
        ])

    # CV parameters
    n_folds = 5
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores_df = pd.DataFrame()

    # With all the features
    print('Getting cv accuracy with all feat...')
    st_time = time.time()

    n_scores = cross_val_score(
        pipe, feat_df_train, Ytrain, scoring='f1_macro', cv=cv, n_jobs=-1)

    cv_scores.append(np.mean(n_scores))
    print('Done in {:.2f} s.'.format(time.time()-st_time))
    cv_scores_df[n_feat_to_select[0]] = n_scores

    # With selected features sets
    for n_feat in n_feat_to_select[1:]:
        print('Getting cv accuracy with n_feat = {}...'.format(n_feat)); st_time = time.time()
        # Defined pipeline that scales, selects n_feat and fits classifier
        # without using any data from the cv-test set
        rfe = RFE(estimator=estimator, n_features_to_select=n_feat, step=0.1)
        rfe_pipe = Pipeline([('scaler', scaler), ('selector', rfe), ('estimator', estimator)])
        # Cross validation
        n_scores = cross_val_score(
            rfe_pipe, feat_df_train[feat_set[-1]], Ytrain, scoring='f1_macro', cv=cv, n_jobs=-1)
        # Append results
        cv_scores.append(np.mean(n_scores))
        cv_scores_df[n_feat] = n_scores
        # Fit RFE and select the most important features
        rfe.fit(feat_df_train[feat_set[-1]], Ytrain)
        feat_set.append(feat_df_train[feat_set[-1]].columns[rfe.support_].to_list())
        print('Done in {:.2f} s.'.format(time.time()-st_time))

    # Get best feature set
    best_n_feat = n_feat_to_select[np.argmax(cv_scores)]
    best_feat_set = feat_set[np.argmax(cv_scores)]

    saving_folder = res_dir + '/seed_'+ str(seed) +  '/RFE/'

    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    pd.Series(best_feat_set).to_csv(saving_folder + '/best_feat_set' + ML_algorithm + '.csv', index=None, header=None)

    # Plot results
    fig, ax = plt.subplots()
    plt.plot(n_feat_to_select, cv_scores)
    plt.title('Feature selection: ' + ML_algorithm + light_type)
    plt.xlabel('N features')
    plt.ylabel('CV F1-score')
    plt.savefig(saving_folder + 'feat_selection_CV_scores_F' + ML_algorithm + light_type + '.png')


    # ==============================================================================
    # Model selection
    # ==============================================================================
    # Use best feature set
    feat_df_train = feat_df_train[best_feat_set]
    feat_df_val = feat_df_val[best_feat_set]

    # Define Parameter Grid

    PARAM_GRID = {
	    'estimator__n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
	    'estimator__max_features': ['sqrt', 'log2'],
	    'estimator__max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
	    'estimator__min_samples_split': [2, 5, 10],
	    'estimator__min_samples_leaf': [1, 2, 4],
	    'estimator__n_jobs': [-1]
	}
    estimator = RandomForestClassifier(random_state=seed)


    # CV parameters
    N_ITER = 20
    n_folds = 5
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scaler = scalingClass(scaling='standardize') # subtracts the mean and divides by the standard deviation

    pipe = Pipeline([
        ('scaler', scaler), ('estimator', estimator)
        ])

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using k fold cross validation,
    # search across N_ITER different combinations.
    print('Getting best model parameters...')
    st_time = time.time()
    model_random = RandomizedSearchCV(
            estimator=pipe, param_distributions=PARAM_GRID, n_iter=N_ITER,
            cv=cv, random_state=seed, refit=True, n_jobs=-1)

    # Fit the random search model
    model_random.fit(feat_df_train, Ytrain)
    print('Done in {:.2f} s.'.format(time.time()-st_time))


    # Save model
    model_save_path = saving_folder + '/fitted_RandomizedSearchCVF' + ML_algorithm + '.p'
    params_save_path = saving_folder + '/fitted_RandomizedSearchCVF' + ML_algorithm + 'best_params_n_feat={}.p'.format(feat_df_train.shape[1])
    pickle.dump(model_random, open(model_save_path, 'wb'))
    pickle.dump(model_random.best_estimator_['estimator'].get_params(), open(params_save_path, 'wb'))

    ## MODEL VALIDATION
    ### Predict on validation samples
    y_pred = model_random.best_estimator_.predict(feat_df_val)
    target_names = ['N2', 'Others']
    report = classification_report(Ytest, y_pred, target_names=target_names)

    with open(saving_folder + 'classification_report' + ML_algorithm + '.txt', 'w') as f:
        f.write(report)
    print(report)

    cf_matrix_val = confusion_matrix(Ytest, y_pred)
    class_names = ('N2', 'others')

    # Create Confusion Matrix and calculate metrics
    print('----------------------------------------')
    print("Confusion Matrix  for validation data")
    print('----------------------------------------')
    dataframe = pd.DataFrame(cf_matrix_val, index=class_names, columns=class_names)
    print(dataframe)

    # Create confmatrix figure and save
    plt.figure(figsize=(8, 8))
    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(saving_folder + 'confmat' + ML_algorithm + light_type + ".jpg", dpi=96)
    # plt.close()

    # ==============================================================================
    # Repurposing Experiment
    # ==============================================================================
    # Make predictions on the repurposing data
    # Impute NaNs and Inf
    repurposing_feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Apply the same imputer as train data
    repurposing_feat_df = pd.DataFrame(imputer.transform(repurposing_feat_df), columns=repurposing_feat_df.columns)
    # Select feats
    repurposing_feat_df = repurposing_feat_df[best_feat_set]
    # Predict the probability of each class
    y_pred_proba =  model_random.best_estimator_.predict_proba(repurposing_feat_df)

    # create a DataFrame for better readability
    prob_df = pd.DataFrame(y_pred_proba, columns=class_names)

    # Combine probabilities with metadata
    # Reset indices to concatenate results without problems
    repurposing_metadata.reset_index(drop=True, inplace=True)
    prob_df.reset_index(drop=True, inplace=True)
    combined_df = pd.concat([repurposing_metadata, prob_df], axis=1)

    # Filter samples that were used during training
    combined_df = combined_df.set_index(['well_name', 'worm_gene', 'drug_type','imgstore'])
    metadata_train = metadata_train.set_index(['well_name', 'worm_gene', 'drug_type','imgstore'])
    # Remove rows of train_df in combined_df
    filtered_df = combined_df[~combined_df.index.isin(metadata_train.index)].reset_index()

    # # to check that training samples where removed
    #print("Original shape:", combined_df.shape)
    #print("Shape after filtering:", filtered_df.shape)

    # Select only the desired columns using loc
    columns_to_keep = ['well_name', 'worm_gene', 'drug_type', 'imgstore','Class', 'N2', 'others']
    filtered_df = filtered_df.loc[:, columns_to_keep]

    # save classification of each well
    filtered_df.to_csv(saving_folder + 'well_classification_probabilities_with_metadata.csv', index=False)
