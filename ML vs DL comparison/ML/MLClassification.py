import os
import numpy as np
from pathlib import Path
import pandas as pd
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.scaling_class import scalingClass
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata
import random
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
import time
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose an ML algorithm')
    parser.add_argument(
        '--ML_algorithm',
        type=str,
        choices=['RF', 'LR', 'XGB'],
        default='RF',
        help='Choose ML algorithm: RF (Random Forest), LR (Logistic Regression), XGB (XGBoost)'
    )
    args = parser.parse_args()

    # Choose ML algorithm
    ML_algorithm = args.ML_algorithm
    print(f"Selected ML algorithm: {ML_algorithm}")
    # ML_algorithm = 'LR'
    # ML_algorithm = 'XGB'
    light_type = 'bluelight'


    # ==============================================================================
    # Data Load and cleaning
    # ==============================================================================
    # Seed for reproducible results
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Paths to dataset files
    proj_root_dir = Path('/mnt/multiwormdataset/DiseaseModels/')
    aux_dir = proj_root_dir / 'AuxiliaryFiles'
    res_dir = '/home/lab/PycharmProjects/MLDiseaseModels/FinalResultsML/' + ML_algorithm + '/'
    analysis_dir = proj_root_dir / 'Analysis'

    feat_file =  proj_root_dir / 'Results/features_summary_tierpsy_plate_20200930_125752.csv'
    fname_file = proj_root_dir / 'Results/filenames_summary_tierpsy_plate_20200930_125752.csv'
    meta_file = aux_dir / 'wells_annotated_metadata.csv'

    # Read hydra metadata
    feat_df, meta_df = read_hydra_metadata(feat_file, fname_file, meta_file)

    # Filter features based on nan ratio values
    max_nan_ratio = 0.05
    feat_df = filter_nan_inf(feat_df, max_nan_ratio, axis=0) # 0 --> filter features

    # Read train and validation dataframes made for the video dataset
    train_csv = pd.read_csv('/home/lab/PycharmProjects/MLDiseaseModels/train.csv')
    val_csv = pd.read_csv('/home/lab/PycharmProjects/MLDiseaseModels/val.csv')

    # Process the dataframe to have the same samples as in the video dataset
    # Create a boolean mask where rows match
    mask_train = meta_df.set_index(['imgstore_name', 'well_name']).index.isin(train_csv.set_index(['File', 'Well']).index)
    mask_val = meta_df.set_index(['imgstore_name', 'well_name']).index.isin(val_csv.set_index(['File', 'Well']).index)

    # Filter out the matching rows
    meta_df_train = meta_df[mask_train]
    feat_df_train = feat_df.loc[meta_df_train.index]

    meta_df_val = meta_df[mask_val]
    feat_df_val = feat_df.loc[meta_df_val.index]

    # Add a column with the classes we defined
    meta_df_train = meta_df_train.copy()
    meta_df_train.loc[:, 'Class'] = meta_df_train['worm_gene'].apply(lambda x: 0 if x == 'N2' else 1)
    Ytrain = meta_df_train.Class

    meta_df_val = meta_df_val.copy()
    meta_df_val.loc[:, 'Class'] = meta_df_val['worm_gene'].apply(lambda x: 0 if x == 'N2' else 1)
    Ytest = meta_df_val.Class


    from collections import Counter
    print("Train set class distribution: ", Counter(Ytrain))
    print("Test set class distribution: ", Counter(Ytest))

    # Impute NaNs and Inf
    feat_df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df_val.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fit imputer only on training data
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(feat_df_train)  # Compute means from training data

    # Apply the same imputer to both train and test data
    feat_df_train = pd.DataFrame(imputer.transform(feat_df_train), columns=feat_df_train.columns)
    feat_df_val = pd.DataFrame(imputer.transform(feat_df_val), columns=feat_df_val.columns)

    # ==============================================================================
    # Feature selection (RFE)
    # ==============================================================================
    if ML_algorithm == 'RF':
        estimator = RandomForestClassifier(random_state=seed)

    elif ML_algorithm == 'LR':
        estimator = LogisticRegression(random_state=seed)

    elif ML_algorithm == 'XGB':
        estimator = XGBClassifier(random_state=seed)

    else:
        print('Error: No valid name')

    n_feat_to_select = [feat_df_train.shape[1]] + [2**i for i in range(11, 6, -1)]
    cv_scores = []
    feat_set = [feat_df_train.columns.to_list()]
    scaler = scalingClass(scaling='standardize')  # subtract mean and divide by std


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

    feat_set_path = saving_folder + '/best_feat_set' + ML_algorithm + '.csv'
    pd.Series(best_feat_set).to_csv(feat_set_path, index=None, header=None)


    fig, ax = plt.subplots()
    plt.plot(n_feat_to_select, cv_scores)
    plt.title('Feature selection: ' + ML_algorithm + light_type)
    plt.xlabel('N features')
    plt.ylabel('CV F1-score')
    plt.savefig(saving_folder + 'feat_selection_CV_scores_F' + ML_algorithm + light_type + '.png')
    plt.close()  # Ensure the plot is closed after showing


    # ==============================================================================
    # Model selection
    # ==============================================================================
    feat_df_train = feat_df_train[best_feat_set]
    feat_df_val = feat_df_val[best_feat_set]


    if ML_algorithm == 'RF':
        PARAM_GRID = {
            'estimator__n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
            'estimator__max_features': ['sqrt', 'log2'],
            'estimator__max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__n_jobs': [-1]
        }
        estimator = RandomForestClassifier(random_state=seed)

    elif ML_algorithm == 'LR':
        PARAM_GRID = {
            'estimator__penalty': ['l1', 'l2'],
            'estimator__C': [10 ** i for i in range(-2, 4)],
            'estimator__solver': ['saga'],
            'estimator__max_iter': [1000]
        }
        estimator = LogisticRegression(random_state=seed)

    elif ML_algorithm == 'XGB':
        PARAM_GRID = {
            'estimator__min_child_weight': [1, 5, 10],
            'estimator__gamma': [0.5, 1, 1.5, 2, 5],
            'estimator__subsample': [0.6, 0.8, 1.0],
            'estimator__colsample_bytree': [0.6, 0.8, 1.0],
            'estimator__max_depth': [3, 4, 5]
        }
        estimator = XGBClassifier(random_state=seed)

    else:
        print('Error: No valid name')


    # CV parameters
    n_folds = 5
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scaler = scalingClass(scaling='standardize') # subtract mean and divide by std

    pipe = Pipeline([
        ('scaler', scaler), ('estimator', estimator)
        ])

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using k fold cross validation,
    # search across N_ITER different combinations.
    N_ITER = 20

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
    pickle.dump(model_random.best_estimator_['estimator'].get_params(),
                open(params_save_path, 'wb'))

    # Test set prediction
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

    plt.figure(figsize=(8, 8))
    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(saving_folder + 'confmat' + ML_algorithm + light_type + ".jpg", dpi=96)
    plt.close()

    ####################
    ## Importance analysis
    ####################
    if ML_algorithm == 'RF' or ML_algorithm =='XGB':
        #Get feature importance scores
        importances = model_random.best_estimator_.named_steps['estimator'].feature_importances_

        # Create a DataFrame for better readability
        feature_names = feat_df_train.columns if isinstance(feat_df_train, pd.DataFrame) else [f'Feature {i}' for i in range(feat_df_train.shape[1])]
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

        # Sort the DataFrame by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        feature_importance_df.to_csv(saving_folder + 'feat_importance.csv')
        # Number of top features to plot
        top_n = 40

        # Sort and get the top N features
        top_features = feature_importance_df.head(top_n)

        # Plot the top N feature importances
        plt.figure(figsize=(16, 6))
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

        # Add spacing to the y-axis labels
        ax = plt.gca()
        ax.set_yticklabels(top_features['Feature'], va='center', fontsize=10)
        ax.yaxis.set_tick_params(pad=10)  # Adjust the padding here

        plt.tight_layout()  # Add spacing to prevent label cutoff
        plt.subplots_adjust(left=0.3)
        # plt.show()
        plt.savefig(saving_folder + 'feat_importance_' + ML_algorithm + light_type + '.png')
