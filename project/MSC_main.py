# from https://github.com/talreiss/Mean-Shifted-Anomaly-Detection/blob/main/main.py
# @article{reiss2021mean,
#   title={Mean-Shifted Contrastive Loss for Anomaly Detection},
#   author={Reiss, Tal and Hoshen, Yedid},
#   journal={arXiv preprint arXiv:2106.03844},
#   year={2021}
# }

import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import MSC_utils as utils
from tqdm import tqdm
import torch.nn.functional as F
import gc
import os
import numpy as np
import umap.umap_ as umap
import json

import matplotlib.pyplot as plt
from torchmetrics.classification import ROC

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_aug, device, args):
    model.eval()
    root_save_folder = np.datetime_as_string(np.datetime64('now')).replace(':','')
    os.makedirs(root_save_folder)
    with open(os.path.join(root_save_folder,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    save_folder = os.path.join(root_save_folder, 'start')
    print('Saving starting results (before training) to folder: {}'.format(save_folder))
    if args.epochs == 0:
        args.max_test_batches = 1e10 # turn off test batch limiting if there is no training
    auc, feature_space, umap_obj = get_score(model, device, train_loader, test_loader, 
                                   args.starting_train_features, max_train_batches=args.max_train_batches, 
                                   max_test_batches=args.max_test_batches, save_folder=save_folder, umap_obj=None)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader_aug, optimizer, center, device, 
                                 args.angular, args.max_train_batches)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        if True: # epoch % 5 == 0: # save results after first epoch and every five epochs
            save_folder = os.path.join(root_save_folder, 'epoch_{}'.format(epoch))
            print('Saving results for epoch to folder: {}'.format(save_folder))
        else:
            save_folder = None # do not save
        # use fitted UMAP for diagnostic plots
        # if epoch < args.epochs-1:
        auc_new, _, _ = get_score(model, device, train_loader, test_loader, max_train_batches=args.max_train_batches, 
                        max_test_batches=args.max_test_batches, save_folder=save_folder, umap_obj=umap_obj)
        if epoch > 0 and auc_new < auc: # reduce learning rate when performance is saturated
            optimizer.lr /= 5.0
        auc = auc_new
        # else:
        #     # do not limit the batches for the last epoch
        #     auc, _, _ = get_score(model, device, train_loader, test_loader, max_train_batches=1e10, 
        #                     max_test_batches=1e10, save_folder=save_folder, umap_obj=umap_obj)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
    
    # save final model to file
    torch.save(model.state_dict(), os.path.join(root_save_folder, 'final_trained_{}.pt'.format(args.backbone)))
    print('Complete.')


def run_epoch(model, train_loader, optimizer, center, device, is_angular, max_train_batches=1e10):
    total_loss, total_num = 0.0, 0
    ct = 0
    for ((img1, img2), _) in tqdm(train_loader, desc='Train...'):
        gc.collect()
        ct += 1
        if ct >= max_train_batches:
            # print('Stopping at max_train_batches')
            break
        img1 = img1.float()
        img2 = img2.float()
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        # total_num += img1.size(0)
        total_num += img1.size(0)
        # total_loss += loss.detach().item() * img1.size(0)
        total_loss += loss.detach().item() * img1.size(0)
        
        gc.collect()

    return total_loss / (total_num)

def save_results_and_diagnostics(save_folder, train_feature_space, test_feature_space, test_images, test_labels, test_distances, umap_obj=None):
    print('Saving results and diagnostics to file...')
    os.makedirs(save_folder)
    
    # project features to 2D using UMAP
    features_concat = np.concatenate((train_feature_space, test_feature_space), axis=0)
    if umap_obj is None:
        umap_obj = umap.UMAP(n_neighbors=20, min_dist=0.2, random_state=2024)
        embedding = umap_obj.fit_transform(features_concat)
    else:
        embedding = umap_obj.transform(features_concat)
    
    # list embedding labels and KNN distances
    n_train = train_feature_space.shape[0]
    n_test = test_feature_space.shape[0]
    embedding_testset = np.concatenate((np.zeros((n_train,)), np.ones((n_test,))))
    embedding_labels = np.concatenate((np.zeros((n_train,)), test_labels))
    train_distances = utils.knn_score(train_feature_space, train_feature_space)
    embedding_distances = np.concatenate((train_distances, test_distances))
    
    # visualize UMAP embeddings by label
    plt.figure()
    idx_plot = embedding_testset == 0
    plt.plot(embedding[idx_plot,0],embedding[idx_plot,1], '.', markersize=5, label='Train (ambient)')
    idx_plot = np.logical_and(embedding_testset==1, embedding_labels==0)
    plt.plot(embedding[idx_plot,0],embedding[idx_plot,1], '.', markersize=5, label='Test (ambient)')
    idx_plot = np.logical_and(embedding_testset==1, embedding_labels==1)
    plt.plot(embedding[idx_plot,0],embedding[idx_plot,1], '.', markersize=18, color='gray')
    plt.plot(embedding[idx_plot,0],embedding[idx_plot,1], '.', markersize=13, label='Test (target)')
    plt.xlabel('UMAP dim 1', fontsize=15)
    plt.ylabel('UMAP dim 2', fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_folder, 'umap_by_label.png'), dpi=500)
    
    # visualize UMAP embeddings by distance
    plt.figure()
    idx_plot = embedding_labels==0
    plt.scatter(embedding[idx_plot,0],embedding[idx_plot,1], 14, embedding_distances[idx_plot], label='Ambient')
    idx_plot = embedding_labels==1
    plt.scatter(embedding[idx_plot,0],embedding[idx_plot,1], 90, embedding_distances[idx_plot], label='Test (target)', 
                edgecolor='gray', linewidth=2)
    plt.xlabel('UMAP dim 1', fontsize=15)
    plt.ylabel('UMAP dim 2', fontsize=15)
    plt.colorbar(label='Distance to train set KNN')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_folder, 'umap_by_distance.png'), dpi=500)
    
    # visualize test set ROC curve
    auc = roc_auc_score(test_labels, test_distances)
    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(torch.Tensor(test_distances), torch.Tensor(test_labels).int())
    plt.figure()
    plt.plot(fpr, tpr, linewidth=3)
    plt.xlabel('FAR', fontsize=15)
    plt.ylabel('PD (Recall)', fontsize=15)
    plt.title('Test set ROC, AUC = {:0.3f}'.format(auc), fontsize=15)
    plt.grid()
    plt.savefig(os.path.join(save_folder, 'roc_curve.png'), dpi=500)
    # adjust axis scale
    plt.figure()
    plt.plot(fpr, tpr, linewidth=3)
    plt.xlabel('FAR', fontsize=15)
    plt.ylabel('PD (Recall)', fontsize=15)
    plt.title('Test set ROC, AUC = {:0.3f}'.format(auc), fontsize=15)
    plt.xscale('log')
    plt.xlim([5e-3, 1])
    plt.grid()
    plt.savefig(os.path.join(save_folder, 'roc_curve_log.png'), dpi=500)
    
    # visualize test set distance distributions
    _, ax = plt.subplots()
    bins = np.linspace(start=np.min(test_distances), stop=np.max(test_distances), num=100)
    plt.hist(test_distances[test_labels==0], bins, density=False, label='Ambient')
    plt.hist(test_distances[test_labels==1], bins, density=False, label='Target')
    plt.xlabel('KNN distance', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title('Test set KNN distances', fontsize=15)
    plt.yscale('log')
    ylim_tmp = ax.get_ylim()
    plt.ylim([0.5, ylim_tmp[-1]])
    plt.grid()
    plt.savefig(os.path.join(save_folder, 'score_dists.png'), dpi=500)
    
    # visualize images of hardest, easiest, and middle-of-the-pack test set
    # note that only some of the images are saved in memory
    test_images = np.array(test_images)
    idx_targets = test_labels[:test_images.shape[0]] == 1
    idx_ambient = test_labels[:test_images.shape[0]] == 0
    test_distances_chop = test_distances[:test_images.shape[0]]
    target_images = test_images[idx_targets,:,:,:]
    target_distances = test_distances_chop[idx_targets]
    ambient_images = test_images[idx_ambient,:,:,:]
    ambient_distances = test_distances_chop[idx_ambient]
    # also plot the embedding locations
    embedding_chop = embedding[n_train:, :]
    embedding_chop = embedding_chop[:test_images.shape[0], :]
    target_embeddings = embedding_chop[idx_targets, :]
    ambient_embeddings = embedding_chop[idx_ambient, :]
    
    # catch failure case without 6 targets or ambient examples
    if len(target_distances) < 6 or len(ambient_distances) < 6:
        print('Fewer than 6 targets or ambient examples, not plotting images')
        return umap_obj
    
    idx_target_sort = np.argsort(target_distances)
    idx_ambient_sort = np.argsort(ambient_distances)
    # visualize hardest and easiest targets
    for ii in [0,1,2,3,4,5, -6,-5,-4,-3,-2,-1]:
        plt.figure()
        idx_plot = idx_target_sort[ii]
        img = np.transpose(target_images[idx_plot,:,:,:],(1,2,0))
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Test Target, KNN distance = {:0.3f}\nUMAP embedding = [{:0.2f}, {:0.2f}]'.format(
            target_distances[idx_plot], target_embeddings[idx_plot,0], target_embeddings[idx_plot,1]))
        plt.savefig(os.path.join(save_folder, 'target_sorted_{}.png'.format(ii)), dpi=500)
    plt.close('all')
    # visualize hardest and easiest ambients
    for ii in [0,1,2,3,4,5, -6,-5,-4,-3,-2,-1]:
        plt.figure()
        idx_plot = idx_ambient_sort[ii]
        img = np.transpose(ambient_images[idx_plot,:,:,:],(1,2,0))
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Test Ambient, KNN distance = {:0.3f}\nUMAP embedding = [{:0.2f}, {:0.2f}]'.format(
            ambient_distances[idx_plot], ambient_embeddings[idx_plot,0], ambient_embeddings[idx_plot,1]))
        plt.savefig(os.path.join(save_folder, 'ambient_sorted_{}.png'.format(ii)), dpi=500)
    plt.close('all')
    # visualize other specific examples
    embedding_point = [3, 6]
    sqdist_to_point = np.square(ambient_embeddings[:,0] - embedding_point[0]) + np.square(ambient_embeddings[:,1] - embedding_point[1])
    idx_ambient_sort = np.argsort(sqdist_to_point)
    for ii in [0,1,2,3,4,5]:
        plt.figure()
        idx_plot = idx_ambient_sort[ii]
        img = np.transpose(ambient_images[idx_plot,:,:,:],(1,2,0))
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Test Ambient, KNN distance = {:0.3f}\nUMAP embedding = [{:0.2f}, {:0.2f}]'.format(
            ambient_distances[idx_plot], ambient_embeddings[idx_plot,0], target_embeddings[idx_plot,1]))
        plt.savefig(os.path.join(save_folder, 'ambient_near_point1_{}.png'.format(ii)), dpi=500)
    plt.close('all')
    embedding_point = [3, 11.5]
    sqdist_to_point = np.square(ambient_embeddings[:,0] - embedding_point[0]) + np.square(ambient_embeddings[:,1] - embedding_point[1])
    idx_ambient_sort = np.argsort(sqdist_to_point)
    for ii in [0,1,2,3,4,5]:
        plt.figure()
        idx_plot = idx_ambient_sort[ii]
        img = np.transpose(ambient_images[idx_plot,:,:,:],(1,2,0))
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Test Ambient, KNN distance = {:0.3f}\nUMAP embedding = [{:0.2f}, {:0.2f}]'.format(
            ambient_distances[idx_plot], ambient_embeddings[idx_plot,0], target_embeddings[idx_plot,1]))
        plt.savefig(os.path.join(save_folder, 'ambient_near_point2_{}.png'.format(ii)), dpi=500)
    plt.close('all')
    
    gc.collect()
    torch.cuda.empty_cache()
    return umap_obj

def get_score(model, device, train_loader, test_loader, starting_train_features=None, 
              max_train_batches=1e10, max_test_batches=1e10, save_folder=None, umap_obj=None):
    gc.collect()
    torch.cuda.empty_cache()
    if starting_train_features is None or not os.path.exists(starting_train_features):
        train_feature_space = []
        with torch.no_grad():
            ct = 0
            for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
                gc.collect()
                ct += 1
                if ct >= max_train_batches:
                    # print('Stopping at max_train_batches')
                    break
                imgs = imgs.float()
                imgs = imgs.to(device)
                features = model(imgs)
                train_feature_space.append(features)
            train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        if starting_train_features is not None:
            print('Saving starting train features to file')
            np.save(starting_train_features, train_feature_space)
    else:
        print('Loading starting train features from file')
        train_feature_space = np.load(starting_train_features)
    test_feature_space = []
    test_labels = []
    if save_folder is not None:
        # save some test images to file
        test_images = []
    with torch.no_grad():
        ct = 0
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            gc.collect()
            ct += 1
            if ct >= max_test_batches:
                # print('Stopping at max_test_batches')
                break
            imgs = imgs.float()
            if save_folder is not None and ct < 200:
                # save some of the test images for later diagnostic plots
                test_images.extend(imgs.numpy())
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    if save_folder is not None:
        umap_obj = save_results_and_diagnostics(save_folder, train_feature_space, test_feature_space, 
                                                test_images, test_labels, distances, umap_obj)

    return auc, train_feature_space, umap_obj

def seed_everything(seed=2024):
#   random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    seed_everything(2024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    model = utils.Model(args.backbone)
    model = model.to(device)

    train_loader, test_loader, train_loader_aug = utils.get_loaders(dataset=args.dataset, 
                                                                  training_data_root=args.training_data_root, 
                                                                  testing_data_root=args.testing_data_root, 
                                                                  label_class=args.label, 
                                                                  batch_size=args.batch_size, 
                                                                  backbone=args.backbone,
                                                                  num_workers=args.num_workers)
    train_model(model, train_loader, test_loader, train_loader_aug, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='xview3', help='cifar10 or xview3')
    parser.add_argument('--training_data_root', default='/mnt/c/Users/titor/Documents/data/xview3/chip_sets/three_channel_safe/validation', help='root directory containing training data')
    parser.add_argument('--testing_data_root', default='/mnt/c/Users/titor/Documents/data/xview3/chip_sets/three_channel_safe/holdout', help='root directory containing test data')
    parser.add_argument('--starting_train_features', default='xview3_vit_starting_train_features_4.npy', help='numpy save file with starting training set features to be loaded or written')
    parser.add_argument('--max_train_batches', default=20, help='Maximum number of batches to iterate over in training')
    parser.add_argument('--max_test_batches', default=20, help='Maximum number of batches to iterate over in testing - not applied at final epoch')
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=5e-5, help='The initial learning rate. Default for cifar10: 1e-5')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--backbone', default='vit_b_16', type=str, help='ResNet18/50/152, convnext_tiny, or vit_b_16')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss?')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of dataloader workers')
    args = parser.parse_args()
    main(args)