# loop through scenes and make chips for training and testing
# training will consist of detections and non-detections (80% split)
# testing will consist of detections and non-detections (20% split)
# Note that for "training" we are currently using the xView3 "validation" set
# because it is better curated, so folders will be marked as "validation" and "holdout"
# and NOT marked as "train" and "test".
# Future work might involve processing the xView3 "training" set.

import pandas
import os
from glob import glob
import numpy as np
from skimage import io
from skimage.util import img_as_uint
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% some configuration

data_dir_root = '/mnt/c/Users/titor/Dropbox/PC/Documents/data/xview3'

figure_dir = '/mnt/c/Users/titor/Documents/data/xview3/figures'

chip_size_pix = 256 # pixels along one edge
target_within_dist_pix = 112 # threshold distance from annotation to chip center to say target is within the chip
target_unsure_dist_pix = 224 # threshold distance from annotation to chip center to say target may be within the chip

dB_thresh = -100 # discard any chips that have any pixels with this low dB intensity or lower

bathy_thresh = -50 # -20 # keep only chips that have this low bathymetry in m or lower

# rescale_limits_dB = [-45, 15] # [-50, 25] # rescale chips to between -1 and 1 based on these assumed overall maximum/minimum

for chip_holdout in [True, False]: # set to True to make chips for the holdout set

    if chip_holdout:
        target_write_dir = os.path.join(data_dir_root,'chip_sets/three_channel_minmax/holdout/target')
        unsure_write_dir = os.path.join(data_dir_root,'chip_sets/three_channel_minmax/holdout/unsure')
        background_write_dir = os.path.join(data_dir_root,'chip_sets/three_channel_minmax/holdout/background')
    else:
        target_write_dir = os.path.join(data_dir_root,'chip_sets/three_channel_minmax/validation/target')
        unsure_write_dir = os.path.join(data_dir_root,'chip_sets/three_channel_minmax/validation/unsure')
        background_write_dir = os.path.join(data_dir_root,'chip_sets/three_channel_minmax/validation/background')
        
    # %% make folders where necessary

    if not os.path.exists(target_write_dir): os.makedirs(target_write_dir)
    if not os.path.exists(unsure_write_dir): os.makedirs(unsure_write_dir)
    if not os.path.exists(background_write_dir): os.makedirs(background_write_dir)

    # %% find all data

    # parse ground truths csv
    gt_csv = os.path.join(data_dir_root, 'validation.csv')
    gt_df = pandas.read_csv(gt_csv)

    # find all scenes VH and VV SAR data and bathymetry data
    if chip_holdout:
        vh_glob = glob(os.path.join(data_dir_root,'scenes_holdout/**/VH_dB.tif'))
    else:
        vh_glob = glob(os.path.join(data_dir_root,'scenes_validation/**/VH_dB.tif'))
    vv_glob = [x.replace('VH_dB', 'VV_dB') for x in vh_glob]
    bathy_glob = [x.replace('VH_dB', 'bathymetry') for x in vh_glob]
    scene_glob = [x.split('/')[-2] for x in vh_glob]
    # find all shoreline data
    shoreline_glob = glob(os.path.join(data_dir_root,'shorelines/*_shoreline.npy'))

    # %% loop over scenes and process
    stats = {'ch0_mean':[],'ch1_mean':[],'ch2_mean':[],'ch0_std':[],'ch1_std':[],'ch2_std':[],}
    for ii in range(len(vh_glob)):
        scene = scene_glob[ii]
        # if os.path.exists(os.path.join(figure_dir,scene+'_vv_vh_bathy.png')):
        #     print('\tWARNING: Skipping processed scene {}'.format(scene))
        #     continue
        
        vh_file = vh_glob[ii]
        vv_file = vv_glob[ii]
        bathy_file = bathy_glob[ii]
        vh_img = io.imread(vh_file)
        vv_img = io.imread(vv_file)
        bathy_img = io.imread(bathy_file)
        
        print('\nProcessing scene {}'.format(scene))
        
        # find corresponding target ground truths
        gt_scene_df = gt_df[gt_df['scene_id'] == scene]
        if len(gt_scene_df.index) == 0:
            print('WARNING: no ground truth targets found for scene {}'.format(scene))
        gt_rows = np.array(gt_scene_df['detect_scene_row'])
        gt_cols = np.array(gt_scene_df['detect_scene_column'])
        max_tb = np.nanmax(np.abs(np.array(gt_scene_df['top'])-np.array(gt_scene_df['bottom'])))
        max_lr = np.nanmax(np.abs(np.array(gt_scene_df['left'])-np.array(gt_scene_df['right'])))
        print('\tMaximum top-bottom target size: {}'.format(max_tb))
        print('\tMaximum left-right target size: {}'.format(max_lr))
        
        if not os.path.exists(os.path.join(figure_dir,scene+'_vv_vh_bathy.png')):
            # find corresponding shoreline file
            shoreline_file = [x for x in shoreline_glob if scene in x]
            assert len(shoreline_file) == 1, 'found no or multiple shoreline files for scene {}'.format(scene)
            shoreline = np.load(shoreline_file[0], allow_pickle=True)
            # for jj in range(len(shoreline)): # shoreline is multiple lines, convert to enclosed polygons
            #     shoreline[jj] = np.concatenate((shoreline[jj],shoreline[jj][0:1,:]), axis=0)

            plt.figure(figsize=(16,5))
            plt.subplot(1,3,1)
            plt.imshow(vh_img[::10,::10])
            for jj in range(len(shoreline)): # shoreline is multiple lines
                plt.plot(shoreline[jj][:,1]/10,shoreline[jj][:,0]/10,'k-')
            plt.colorbar(label='dB')
            plt.clim(-60,0)
            plt.title(scene + ', VH polarization')
            plt.subplot(1,3,2)
            plt.imshow(vv_img[::10,::10])
            for jj in range(len(shoreline)): # shoreline is multiple lines
                plt.plot(shoreline[jj][:,1]/10,shoreline[jj][:,0]/10,'k-')
            plt.colorbar(label='dB')
            plt.clim(-60,0)
            plt.title(scene + ', VV polarization')
            plt.subplot(1,3,3)
            plt.imshow(bathy_img, cmap='bwr')
            plt.colorbar(label='m')
            plt.clim(-90,90)
            plt.title(scene + ', bathymetry')
            # plt.show()
            plt.savefig(os.path.join(figure_dir,scene+'_vv_vh_bathy.png'))
            plt.close()
        
        # choose chip centerpoints
        rows, cols = vh_img.shape
        assert (rows, cols) == vv_img.shape, 'vv and vh images have different sizes unexpectedly'
        half_chip = int(np.floor(chip_size_pix/2))
        chip_centers_row = np.array(range(half_chip,rows-half_chip,chip_size_pix))
        chip_edges_row = np.array(range(0,rows,chip_size_pix))
        chip_centers_col = np.array(range(half_chip,cols-half_chip,chip_size_pix))
        chip_edges_col = np.array(range(0,cols,chip_size_pix))
        
        N_chips_pre_filt = len(chip_centers_row) * len(chip_centers_col)
        print('\t{} possible chips before any filtering'.format(N_chips_pre_filt))
        
        rows_bathy, cols_bathy = bathy_img.shape
        
        # loop over chips
        N_chips_rejected_bathy = 0
        N_chips_rejected_intensity = 0
        N_target_chips = 0
        N_unsure_chips = 0
        N_bg_chips = 0
        for jj in tqdm(range(len(chip_centers_row)), desc='Processing chips'):
            for kk in range(len(chip_centers_col)):
                # do simple nearest neighbor interpolation to find bathymetry at chip center
                center_perc_row = float(chip_centers_row[jj]) / float(rows)
                center_perc_col = float(chip_centers_col[kk]) / float(cols)
                bathy_chip = bathy_img[round(center_perc_row*rows_bathy),round(center_perc_col*cols_bathy)]
                # reject if bathy is above the threshold or nearby bathy pixels are above the threshold
                if bathy_chip > bathy_thresh:
                    N_chips_rejected_bathy += 1
                    continue
                bathy_chip_surround = bathy_img[max(round(center_perc_row*rows_bathy)-4, 0):min(round(center_perc_row*rows_bathy)+5, rows_bathy),
                                                max(round(center_perc_col*cols_bathy)-4, 0):min(round(center_perc_col*cols_bathy)+5, cols_bathy),]
                if any(bathy_chip_surround.flatten() > bathy_thresh):
                    N_chips_rejected_bathy += 1
                    continue
                
                # construct chip as three channels: vh and vv, and mean of the two
                chip = np.zeros((chip_size_pix, chip_size_pix, 3),dtype=np.float16)
                chip[:,:,0] = vh_img[chip_edges_row[jj]:chip_edges_row[jj+1], chip_edges_col[kk]:chip_edges_col[kk+1]]
                chip[:,:,2] = vv_img[chip_edges_row[jj]:chip_edges_row[jj+1], chip_edges_col[kk]:chip_edges_col[kk+1]]
                chip[:,:,1] = np.mean(chip[:,:,::2], axis=-1)
                # chip = np.mean(chip, axis=-1)
                if np.any(chip < dB_thresh):
                    N_chips_rejected_intensity += 1
                    continue
                
                # # rescale to between 0 and 1 using global min and max
                # chip -= rescale_limits_dB[0]
                # chip = chip / (rescale_limits_dB[1] - rescale_limits_dB[0])
                # chip = np.clip(chip,0,1)
                
                # rescale to between 0 and 1
                rescale_limits_perchip = [np.min(chip.flatten()), np.max(chip.flatten())]
                chip -= rescale_limits_perchip[0]
                chip = chip / (rescale_limits_perchip[1] - rescale_limits_perchip[0])
                chip = np.clip(chip,0,1)
                
                # test for target, unsure, or background label
                if np.any(np.logical_and(np.abs(gt_rows - chip_centers_row[jj]) < target_within_dist_pix, 
                                        np.abs(gt_cols - chip_centers_col[kk]) < target_within_dist_pix)):
                    label = 'target'
                    chip_write_dir = target_write_dir
                    N_target_chips += 1
                elif np.any(np.logical_and(np.abs(gt_rows - chip_centers_row[jj]) < target_unsure_dist_pix, 
                                        np.abs(gt_cols - chip_centers_col[kk]) < target_unsure_dist_pix)):
                    label = 'unsure'
                    chip_write_dir = unsure_write_dir
                    N_unsure_chips += 1
                else:
                    label = 'background'
                    chip_write_dir = background_write_dir
                    N_bg_chips += 1
                    # catalogue chip statistics
                    stats['ch0_mean'].append(np.mean(chip[:,:,0].flatten()))
                    stats['ch1_mean'].append(np.mean(chip[:,:,1].flatten()))
                    stats['ch2_mean'].append(np.mean(chip[:,:,2].flatten()))
                    stats['ch0_std'].append(np.std(chip[:,:,0].flatten()))
                    stats['ch1_std'].append(np.std(chip[:,:,1].flatten()))
                    stats['ch2_std'].append(np.std(chip[:,:,2].flatten()))
                
                chip_name = '{}_{}_{}_{}.tiff'.format(scene,chip_centers_row[jj],chip_centers_col[kk],label)
                chip_file = os.path.join(chip_write_dir, chip_name)
                # plt.hist(np.ndarray.flatten(chip)), plt.show()
                io.imsave(chip_file, img_as_uint(np.array(chip,dtype='float64')))
                
        print('\t{} chips removed for bathymetry'.format(N_chips_rejected_bathy))
        print('\t{} chips removed for intensity'.format(N_chips_rejected_intensity))
        print('\t{} target chips'.format(N_target_chips))
        print('\t{} unsure chips'.format(N_unsure_chips))
        print('\t{} background chips'.format(N_bg_chips))
        print('')

    print('Overall background chip stats:')
    stats_df = pandas.DataFrame.from_dict(stats)
    if chip_holdout:
        stats_df.to_csv('three_channel_minmax_stats_holdout_20241128.csv')
    else:
        stats_df.to_csv('three_channel_minmax_stats_validation_20241128.csv')
    print('\tCh0 mean: {}, avg std: {}'.format(np.mean(stats['ch0_mean']), np.mean(stats['ch0_std'])))
    print('\tCh1 mean: {}, avg std: {}'.format(np.mean(stats['ch1_mean']), np.mean(stats['ch1_std'])))
    print('\tCh2 mean: {}, avg std: {}'.format(np.mean(stats['ch2_mean']), np.mean(stats['ch2_std'])))


    print('Complete.')