# load target and background chips, compute some basic stats, save to file, and visualize

import pandas as pd
import os
from glob import glob
import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% some configuration

process_holdout = True

if process_holdout:
    data_dir_root = '/mnt/c/Users/titor/Dropbox/PC/Documents/data/xview3/chip_sets/holdout'
    figure_dir = '/mnt/c/Users/titor/Dropbox/PC/Documents/data/xview3/figures/chip_stats/holdout'
else:
    data_dir_root = '/mnt/c/Users/titor/Dropbox/PC/Documents/data/xview3/chip_sets/validation'
    figure_dir = '/mnt/c/Users/titor/Dropbox/PC/Documents/data/xview3/figures/chip_stats'

csv_save_dir = '/mnt/c/Users/titor/Dropbox/Projects/Anomaly_detection/explore-anomaly/analysis'

# %% find target chips

target_glob = glob(os.path.join(data_dir_root,'target/*.tif'))
N_targets = len(target_glob)

# %% loop through and process targets

target_stats_df = pd.DataFrame.from_dict({'chip_file':np.empty((N_targets,)),
                                          'mean':np.full((N_targets,),np.nan),
                                          'stdv':np.full((N_targets,),np.nan),
                                          'min':np.full((N_targets,),np.nan),
                                          'max':np.full((N_targets,),np.nan),
                                          'range':np.full((N_targets,),np.nan),
                                          'max_gradient':np.full((N_targets,),np.nan),
                                          'min_gradient':np.full((N_targets,),np.nan)})
for ii, chip_file in tqdm(enumerate(target_glob), desc='Processing target chips', total=len(target_glob)):
    chip = np.array(io.imread(chip_file), dtype=float)
    
    # compute some basic stats
    target_stats_df['mean'].iloc[ii] = np.mean(chip)
    target_stats_df['stdv'].iloc[ii] = np.std(chip)
    target_stats_df['min'].iloc[ii] = np.min(chip)
    target_stats_df['max'].iloc[ii] = np.max(chip)
    target_stats_df['range'].iloc[ii] = np.max(chip) - np.min(chip)
    
    # downsample and compute largest gradient
    chip_downsample = transform.resize(chip, (56, 56), anti_aliasing=True)
    chip_grad_x = np.gradient(chip_downsample, axis=0)
    chip_grad_y = np.gradient(chip_downsample, axis=1)
    target_stats_df['max_gradient'].iloc[ii] = np.max([np.max(chip_grad_x), np.max(chip_grad_y)])
    target_stats_df['min_gradient'].iloc[ii] = np.min([np.min(chip_grad_x), np.min(chip_grad_y)])
    
    if ii < 10:
        plt.figure(), plt.imshow(chip), plt.savefig(os.path.join(figure_dir,'target_example_{}.png'.format(ii))), plt.close()
        plt.figure(), plt.imshow(chip_downsample), plt.savefig(os.path.join(figure_dir,'target_downsampled_example_{}.png'.format(ii))), plt.close()

target_stats_df['chip_file'] = target_glob

if process_holdout:
    target_stats_df.to_csv(os.path.join(csv_save_dir, 'holdout_target_chip_stats.csv'))
else:
    target_stats_df.to_csv(os.path.join(csv_save_dir, 'validation_target_chip_stats.csv'))


# %% find background chips

background_glob = glob(os.path.join(data_dir_root,'background/*.tif'))
N_targets = len(background_glob)

# %% loop through and process background

background_stats_df = pd.DataFrame.from_dict({'chip_file':np.empty((N_targets,)),
                                          'mean':np.full((N_targets,),np.nan),
                                          'stdv':np.full((N_targets,),np.nan),
                                          'min':np.full((N_targets,),np.nan),
                                          'max':np.full((N_targets,),np.nan),
                                          'range':np.full((N_targets,),np.nan),
                                          'max_gradient':np.full((N_targets,),np.nan),
                                          'min_gradient':np.full((N_targets,),np.nan)})
for ii, chip_file in tqdm(enumerate(background_glob), desc='Processing background chips', total=len(background_glob)):
    chip = np.array(io.imread(chip_file), dtype=float)
    
    # compute some basic stats
    background_stats_df['mean'].iloc[ii] = np.mean(chip)
    background_stats_df['stdv'].iloc[ii] = np.std(chip)
    background_stats_df['min'].iloc[ii] = np.min(chip)
    background_stats_df['max'].iloc[ii] = np.max(chip)
    background_stats_df['range'].iloc[ii] = np.max(chip) - np.min(chip)
    
    # downsample and compute largest gradient
    chip_downsample = transform.resize(chip, (56, 56), anti_aliasing=True)
    chip_grad_x = np.gradient(chip_downsample, axis=0)
    chip_grad_y = np.gradient(chip_downsample, axis=1)
    background_stats_df['max_gradient'].iloc[ii] = np.max([np.max(chip_grad_x), np.max(chip_grad_y)])
    background_stats_df['min_gradient'].iloc[ii] = np.min([np.min(chip_grad_x), np.min(chip_grad_y)])
    
    if ii < 10:
        plt.figure(), plt.imshow(chip), plt.savefig(os.path.join(figure_dir,'background_example_{}.png'.format(ii))), plt.close()
        plt.figure(), plt.imshow(chip_downsample), plt.savefig(os.path.join(figure_dir,'background_downsampled_example_{}.png'.format(ii))), plt.close()

background_stats_df['chip_file'] = background_glob

if process_holdout:
    background_stats_df.to_csv(os.path.join(csv_save_dir, 'holdout_background_chip_stats.csv'))
else:
    background_stats_df.to_csv(os.path.join(csv_save_dir, 'validation_background_chip_stats.csv'))


print('Complete.')