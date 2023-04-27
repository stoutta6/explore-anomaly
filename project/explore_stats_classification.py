# view chip statistics and explore KNN classification method

import pandas as pd
import os
from glob import glob
import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import sweetviz as sv
from sklearn import metrics

# %% some configuration

figure_dir = '//mnt/c/Users/titor/Dropbox/Projects/Anomaly_detection/explore-anomaly/figures/analysis'
csv_save_dir = '/mnt/c/Users/titor/Dropbox/Projects/Anomaly_detection/explore-anomaly/analysis'

run_reports = False

KNN_K = 10 # average distance over this many nearest neighbors as an anomaly criterion

# %% load all data

validation_target_df = pd.read_csv(os.path.join(csv_save_dir,'validation_target_chip_stats.csv'))
validation_background_df = pd.read_csv(os.path.join(csv_save_dir,'validation_background_chip_stats.csv'))
holdout_target_df = pd.read_csv(os.path.join(csv_save_dir,'holdout_target_chip_stats.csv'))
holdout_background_df = pd.read_csv(os.path.join(csv_save_dir,'holdout_background_chip_stats.csv'))

# %% concatenate validation data and view report

N_val_targets = len(validation_target_df.index)
validation_target_df['label'] = ['target'] * N_val_targets
N_val_background = len(validation_background_df.index)
validation_background_df['label'] = ['background'] * N_val_background

validation_df = pd.concat([validation_target_df, validation_background_df])

if run_reports:
    my_report = sv.analyze(validation_df)
    my_report.show_html('analysis/validation__concat_sweetviz_report.html')
    my_report = sv.compare([validation_target_df, 'Target'], [validation_background_df, 'Background'])
    my_report.show_html('analysis/validation_comparison_sweetviz_report.html')

N_holdout_targets = len(holdout_target_df.index)
holdout_target_df['label'] = ['target'] * N_holdout_targets
N_holdout_background = len(holdout_background_df.index)
holdout_background_df['label'] = ['background'] * N_holdout_background

holdout_df = pd.concat([holdout_target_df, holdout_background_df])

if run_reports:
    my_report = sv.analyze(holdout_df)
    my_report.show_html('analysis/holdout_concat_sweetviz_report.html')
    my_report = sv.compare([holdout_target_df, 'Target'], [holdout_background_df, 'Background'])
    my_report.show_html('analysis/holdout_comparison_sweetviz_report.html')


# %% Try KNN classification, training only with background examples

features = np.array([validation_background_df['max'], validation_background_df['max_gradient']])
# normalize
features_mean = np.full((len(features),),np.nan)
features_stdv = np.full((len(features),),np.nan)
for ii in range(len(features)):
    features_mean[ii] = np.mean(features[ii,:])
    features_stdv[ii] = np.std(features[ii,:])
    features[ii,:] = features[ii,:] - np.mean(features[ii,:])
    features[ii,:] = features[ii,:] / np.std(features[ii,:])

# apply normalization to holdouts
holdout_features = np.array([holdout_df['max'], holdout_df['max_gradient']])
for ii in range(len(holdout_features)):
    holdout_features[ii,:] = holdout_features[ii,:] - features_mean[ii]
    holdout_features[ii,:] = holdout_features[ii,:] / features_stdv[ii]

# compute distance to KNN
anomaly_score = np.full((len(holdout_features[0]),),np.nan)
for ii in tqdm(range(len(anomaly_score)), desc='Computing anomaly scores', total=len(anomaly_score)):
    distances_to_train = np.full((len(features[0]),),0.0)
    for kk in range(len(features)):
        distances_to_train += np.square(holdout_features[kk,ii] - features[kk,:])
    distances_to_train = np.sort(np.sqrt(distances_to_train))
    anomaly_score[ii] = np.mean(distances_to_train[0:KNN_K])

# %% plot performance

plt.figure()
plt.hist([anomaly_score[holdout_df['label'] == 'background'], anomaly_score[holdout_df['label'] == 'target']])
plt.xlabel('Anomaly score')
plt.ylabel('Count')
plt.yscale('log')
plt.legend(['Background', 'Target'])
plt.title('Anomaly detection by KNN, K = {}'.format(KNN_K))
plt.grid()
plt.savefig('analysis/KNN_holdout_scores_histogram.png')

fpr, tpr, thresholds = metrics.roc_curve(holdout_df['label'], anomaly_score, pos_label='target')

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Anomaly detection task')
plt.grid()
plt.savefig('analysis/KNN_holdout_ROC.png')

print('Complete.')