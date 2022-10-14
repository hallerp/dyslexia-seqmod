#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SVM Baseline with RFE
# Autorin: Sarah Kiener
# Matrikelnr.: 09-110-958
# Datum: 08.12.2021


import pandas as pd
import numpy as np
import math
import random
import argparse

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from datetime import datetime


parser = argparse.ArgumentParser(description="Run Chinese Dyslexia Baseline")
# TODO: Add after implementing random forest
# parser.add_argument("--model", dest="model")
parser.add_argument("--subjpred", dest="batch_subjects", action="store_true")
parser.add_argument("--textpred", dest="batch_subjects", action="store_false")
parser.set_defaults(batch_subjects=True)
args = parser.parse_args()

np.random.seed(42)
random.seed(42)

NUM_FOLDS = 10
BATCH_SUBJECTS = args.batch_subjects
EXCLUDE_MISSING = False


def clean_data(sacc_file):
    if BATCH_SUBJECTS:
        METADATA = ['trial', 'item', 'sn']
    else:
        METADATA = ['trial', 'item']
    CHINESE_DATA = ['wl.c', 'stroke', 'freq', 'freq.c', 'st', 'wn']
    REDUNDANT_DATA = ['GROUP', 'blink', 'firstLast', 'blinkidx', 'WL', 'FT', 'refix', 'nfix', 'wl', 'Unnamed: 0', "nw", "skip", "isamp", "ls", "ft", "sfix", "nf"]

    # drop redundant and Chinese data and metadata
    data_df = sacc_file.drop(REDUNDANT_DATA, axis=1)
    data_df = data_df.drop(CHINESE_DATA, axis=1)
    data_df = data_df.drop(METADATA, axis=1)

    # add log transformed gaze durations
    data_df['gz'] = data_df['gaze'].map(lambda x: math.log(x) if x >= 1 else 0)
    data_df = data_df.drop("gaze", axis=1)

    # convert group labels to 0 (control) and 1 (dyslexic)
    data_df['group'] = data_df['group'].map(lambda x: int(x + 0.5))

    return data_df


def aggregate_data(data_df):
    # group data by subject
    if BATCH_SUBJECTS:
        grouped_data = data_df.groupby('subj')
    else:
        grouped_data = data_df.groupby(['subj', 'sn'])

    # calculate means and standard deviations of all ET features
    means = grouped_data.mean()
    stds = grouped_data.std()

    # join the means and standard deviations in one dataframe
    feature_group_df = means.join(stds, lsuffix="_mean", rsuffix="_std")

    return feature_group_df


def dataframe_to_vecs(data_df, folds):
    feature_folds = []
    label_folds = []
    # get subj column
    if BATCH_SUBJECTS:
        data_df['subj'] = data_df.index
    else:
        all_subj_sent = data_df.index.to_numpy()
        data_df['subj'] = [subj_sent[0] for subj_sent in all_subj_sent]
    # drop the group_mean and the group_std from the features
    # feature_df = data_df.drop(['group_mean', 'group_std'], axis=1)
    for _, fold in enumerate(folds):
        fold_data = data_df[data_df['subj'].isin(fold)]
        feature_folds.append(
            fold_data.drop(['group_mean', 'group_std', 'subj'], axis=1).to_numpy()
        )
        label_folds.append(fold_data[['group_mean']].to_numpy().transpose().flatten())
    return feature_folds, label_folds


#def shuffle(norm_feature_vec, label_vec):
#    idx = np.random.permutation(len(norm_feature_vec))
#    norm_feature_vec, label_vec = norm_feature_vec[idx], label_vec[idx]
#    return norm_feature_vec, label_vec


def get_train_dev_test_splits(n_folds, i_test, j_dev, feature_splits, label_splits):
    counter = 0
    test_features = feature_splits[i_test]
    test_labels = label_splits[i_test]

    dev_features = feature_splits[j_dev]
    dev_labels = label_splits[j_dev]

    train_labels = np.array([])
    for idx, elem in enumerate(feature_splits):
        if idx != i_test and idx != j_dev:
            if counter == 0:
                train_features = elem
                counter += 1
            else:
                train_features = np.concatenate((train_features, elem), axis=0)

    for idx, elem in enumerate(label_splits):
        if idx != i_test and idx != j_dev:
            train_labels = np.concatenate((train_labels, elem), axis=0)

    return train_features, dev_features, test_features, train_labels, dev_labels, test_labels


def svm_model(train_features, train_labels, param):
    classifier = svm.SVC(kernel='linear', C=param, probability=False)
    classifier.fit(train_features, train_labels)
    return classifier


def rfe_model(train_features, train_labels, param, n_features):
    # define the classifier model
    classifier = svm.SVC(kernel='linear', C=param, probability=False)

    # recursive feature selection
    selector = RFE(classifier, n_features_to_select=n_features, step=1)
    selector = selector.fit(train_features, train_labels)

    return selector


def evaluate(classifier, features, labels):
    pred = classifier.predict(features)
    accuracy = accuracy_score(labels, pred)
    # specificity = tn / (tn + fp)
    # specificity = specificity_score(labels, pred)
    # recall: tp / (tp + fn)
    recall = recall_score(labels, pred)
    # presicion = tp / (tp + fp)
    precision = precision_score(labels, pred)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(labels, pred)

    return [accuracy, precision, recall, f1]


def compare_hyperparameters(train_features, train_labels, dev_features, dev_labels, c_params, n_features): 
    # array that collect the scores for each param and each number of features
    eval_scores_all_params = np.zeros([len(c_params), n_features, 4])
    
    # for each parameter and each number of feature train a classifier
    for param in c_params:
        for feature in range(n_features, 0, -1):
            rfe = rfe_model(train_features, train_labels, param, feature)
            # evaluate the classifier on the dev set
            scores_per_feature = evaluate(rfe, dev_features, dev_labels)
            # include the scores in the score array
            eval_scores_all_params[c_params.index(param), feature-1] += scores_per_feature
        
    return np.array(eval_scores_all_params)


def nested_cv_rfe(feature_folds, label_folds, n_folds, c_params, n_features):
    # array to collect the scores during training
    scores = np.zeros([n_folds, len(c_params), n_features, 4])
    scores_per_feature = np.zeros([n_folds, n_features, 4])
    # lists to collect the best params and the best number of features per fold and the final evaluations per fold
    best_params = []
    best_n_features = []
    CV_eval = []

    # set index for test set
    for i_test in range(n_folds):
        # set index for dev set
        for j_dev in range(n_folds):
            # try each combination of test and dev sets
            if i_test != j_dev:
                train_features, dev_features, test_features, train_labels, dev_labels, test_labels = get_train_dev_test_splits(
                    n_folds, i_test, j_dev, feature_folds, label_folds)
                scaler = MinMaxScaler()
                train_features = scaler.fit_transform(train_features)
                dev_features = scaler.transform(dev_features)
                # compare the 8 hyperparameters on each test - dev combination
                scores[i_test] += compare_hyperparameters(train_features, train_labels, dev_features, dev_labels,
                                                          c_params, n_features)

        # calculate the mean of evaluations for each parameter over all n_folds-1 dev sets
        # (-1, because one fold is always excluded as test set)
        scores[i_test] /= (n_folds - 1)

         # for each fold, extract the index of the best parameter and best number of features regarding f1-score
        f1 = scores[i_test, :, :, 3]
        # returns a tuple with indices (row, column) for the max value: row corresponds to param, column to n_features
        indices = np.unravel_index(np.argmax(f1), f1.shape)
        # retrieve best parameter from list via its index
        best_param = c_params[indices[0]]
        best_params.append(best_param)
        # retrieve best number of features by adding 1 to its index (since index starts at 0)
        best_n_feature = indices[1] + 1
        best_n_features.append(best_n_feature)

        # train RFE model using best hyperparameter from inner cross-validation, on 90% train data (no dev set)
        train_features_fold, dev_features_fold, test_features_fold, train_labels_fold, dev_labels_fold, test_labels_fold = get_train_dev_test_splits(
            n_folds, i_test, i_test, feature_folds, label_folds)
        scaler = MinMaxScaler()
        train_features_fold = scaler.fit_transform(train_features_fold)
        test_features_fold = scaler.transform(test_features_fold)

        final_model = rfe_model(train_features_fold, train_labels_fold, best_param, best_n_feature)
        test_scores = evaluate(final_model, test_features_fold, test_labels_fold)
        viz = RocCurveDisplay.from_estimator(
            final_model,
            test_features_fold,
            test_labels_fold,
            name="ROC fold {}".format(i_test + 1),
            alpha=0.3,
            lw=1,
            ax=ax,
            drop_intermediate=False
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # append the best scores per fold to the final evaluation list
        #scores_per_fold = scores_per_feature[i_test, pos_feature]
        # CV_eval += [scores_per_fold]
        CV_eval += [test_scores]

    # calcualte the mean evaluation scores over all n_folds
    final_scores_mean = np.mean(CV_eval, axis=0)
    final_scores_std = np.std(CV_eval, axis=0)
    print('final scores mean :', final_scores_mean)
    print('final scores sd: ', final_scores_std)
    return final_scores_mean, final_scores_std, best_params, best_n_features, CV_eval



tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# read in the data file
sacc_file = pd.read_csv('data/saccades.csv')
stimuli = pd.read_excel("data/Pan_et_al_2014_VR.xlsx")
if EXCLUDE_MISSING:
    rows_before = len(sacc_file)
    sacc_file = sacc_file[sacc_file["sn"].isin(stimuli.SN.unique())]
    rows_after = len(sacc_file)
    print(f"Dropped {rows_before - rows_after} rows due to missing stimulus data")
# clean the data
data_df = clean_data(sacc_file)

dyslexic_subjects = data_df[data_df["group"] == 1]["subj"].unique()
control_subjects = data_df[data_df["group"] == 0]["subj"].unique()

# aggregate data: calculate mean and std for each ET measure
feature_group_df = aggregate_data(data_df)

# shuffle and distribute on stratified folds
n_folds = NUM_FOLDS
folds = [[] for _ in range(n_folds)]
random.shuffle(dyslexic_subjects)
random.shuffle(control_subjects)
for i, subj in enumerate(dyslexic_subjects):
    folds[i % n_folds].append(subj)
for i, subj in enumerate(control_subjects):
    folds[n_folds - 1 - i % n_folds].append(subj)
for fold in folds:
    random.shuffle(fold)


# vectorize labels and features
feature_folds, label_folds = dataframe_to_vecs(feature_group_df, folds)
# define hyperparams and number of features to test

n_features = 40
c_params = [0.1, 0.5, 1, 5, 10, 50, 100, 500]

# run cross validation with recursive feature selection
final_scores, final_scores_std, best_params, best_n_features, CV_eval = nested_cv_rfe(feature_folds, label_folds, n_folds, c_params,
                                                                    n_features)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
# print('mean auc :', mean_auc)
std_auc = np.std(aucs)
print("auc:", np.mean(aucs), np.std(aucs))
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC Baseline",
)
ax.legend(loc="lower right")
date = datetime.now().strftime('%Y%m%d-%H:%M')
pred_level = 'subjectpred' if BATCH_SUBJECTS else 'textpred'
plt.savefig(f"roc_baseline_{pred_level}_{date}.png", bbox_inches='tight', dpi=500)


final_scores = np.insert(final_scores, 0,  np.mean(aucs), axis=0)
final_scores_std = np.insert(final_scores_std, 0, np.std(aucs), axis=0)
out_str = ""
with open(f"baseline_scores_{pred_level}.txt", 'w') as f:
    for i in range(len(final_scores)):
        out_str += f"${round(final_scores[i],2):1.2f}\db{{{round(final_scores_std[i],2):1.2f}}}$"
        if i < len(final_scores)-1:
            out_str += " & "
        else:
            out_str += " \\\\ "
    f.write(out_str)
