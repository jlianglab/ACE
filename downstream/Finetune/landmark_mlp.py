import numpy as np
from os.path import isfile, join
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
from einops import rearrange
from torchvision import transforms
import os
import csv
from models.swin_transformer import SwinTransformer
import torch.nn as nn
from utils import save_model_wo_conf
from timm.utils import AverageMeter
from torch import optim as optim
import torch.backends.cudnn as cudnn
import time
import math
import sys
import copy
import argparse




class Landmark_Classfication(Dataset):
    def __init__(self, imageEmbeddingDirectory,test, total_folds =5, fold=1):
        self.test = test
        image_embedding = np.load(imageEmbeddingDirectory,allow_pickle=True).item()
        img_list = list(image_embedding.keys())
        fold_size = len(img_list) // total_folds
        folds = []
        for i in range(total_folds):
            start = i * fold_size
            end = (i + 1) * fold_size
            folds.append(img_list[start:end])
        train_images = []
        for i in range(total_folds):
            if i == fold - 1:
                test_images = folds[i]
            else:
                train_images = train_images + folds[i]
        self.data_pairs = []
        if self.test:
            for image_name in test_images:
                self.data_pairs = self.data_pairs + image_embedding[image_name]
        else:
            for image_name in train_images:
                self.data_pairs = self.data_pairs + image_embedding[image_name]


    def __getitem__(self, index):

        image_embedding, image_label = self.data_pairs[index]

        return torch.from_numpy(image_embedding), image_label.long()

    def __len__(self):
        return len(self.data_pairs)




class Landmark_Classfication_Two_Models(Dataset):
    def __init__(self, anatomyEmbeddingDirectory, diseaseEmbeddingDirectory,test, total_folds =5, fold=1, feature_merging="sum"):
        self.test = test
        self.feature_merging = feature_merging
        anatomy_image_embedding = np.load(anatomyEmbeddingDirectory,allow_pickle=True).item()

        disease_image_embedding = np.load(diseaseEmbeddingDirectory,allow_pickle=True).item()


        img_list = list(anatomy_image_embedding.keys())
        fold_size = len(img_list) // total_folds
        folds = []
        for i in range(total_folds):
            start = i * fold_size
            end = (i + 1) * fold_size
            folds.append(img_list[start:end])
        train_images = []
        for i in range(total_folds):
            if i == fold - 1:
                test_images = folds[i]
            else:
                train_images = train_images + folds[i]
        self.anatomy_data_pairs = []
        self.disease_data_pairs = []

        if self.test:
            for image_name in test_images:
                self.anatomy_data_pairs = self.anatomy_data_pairs + anatomy_image_embedding[image_name]
                self.disease_data_pairs = self.disease_data_pairs + disease_image_embedding[image_name]


        else:
            for image_name in train_images:
                self.anatomy_data_pairs = self.anatomy_data_pairs + anatomy_image_embedding[image_name]
                self.disease_data_pairs = self.disease_data_pairs + disease_image_embedding[image_name]

    def __getitem__(self, index):

        anatomy_image_embedding, anatomy_image_label = self.anatomy_data_pairs[index]

        disease_image_embedding, disease_image_label = self.disease_data_pairs[index]



        if self.feature_merging == "sum":
            return torch.from_numpy(anatomy_image_embedding+disease_image_embedding), anatomy_image_label.long()
        elif self.feature_merging == "average":
            return torch.from_numpy(anatomy_image_embedding+disease_image_embedding)/2, anatomy_image_label.long()
        elif self.feature_merging == "concat":

            return torch.concat([torch.from_numpy(anatomy_image_embedding), torch.from_numpy(disease_image_embedding)]), anatomy_image_label.long()
    def __len__(self):
        return len(self.anatomy_data_pairs)








class MLP(nn.Module):
    def __init__(self, num_classes, mlp_in_feature):
        super().__init__()

        self.linear = torch.nn.Linear(mlp_in_feature, 1376)
        self.cls_head = torch.nn.Sequential(torch.nn.Linear(1376, 512),
                                    torch.nn.BatchNorm1d(512, affine=False, eps=1e-6),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(512, 256),
                                    torch.nn.BatchNorm1d(256, affine=False, eps=1e-6),
                                    torch.nn.ReLU(inplace=True),
                                    torch.nn.Linear(256, num_classes))


    def forward(self, embedding):
        return self.cls_head(self.linear(embedding))






def train_one_epoch(mlp ,criterion, optimizer, train_loader, epoch):

    mlp.train(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for idx, (embedding, lbl) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = embedding.shape[0]
        embedding = embedding.float().cuda(non_blocking=True)
        lbl = lbl.cuda(non_blocking=True)
        outputs = mlp(embedding)
        loss = criterion(outputs,lbl)


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
            # update metric
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if (idx + 1) % 10 == 0:

            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, lr=optimizer.param_groups[0]['lr'], ttloss=losses))

            sys.stdout.flush()

    return losses.avg




def test_acc(mlp, test_loader):
    mlp.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (embedding, lbl) in enumerate(test_loader):
            embedding = embedding.cuda(non_blocking=True)
            lbl = lbl.cuda(non_blocking=True)
            out = mlp(embedding)
            out = torch.softmax(out, dim=1)
            _, predicted = torch.max(out, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()

    return 100 * correct / total



def train_loops(args, dataset_name, pretrain_mode, fold=1):
    '''
    fold: 1,2,3,4,5. Which fold to test
    '''


    if dataset_name == "9landmark_classfication":
        num_classes = 9
        train_dataset = Landmark_Classfication("{}/{}/{}_9landmarks.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=False,fold=int(fold))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)


        test_dataset = Landmark_Classfication("{}/{}/{}_9landmarks.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=True,fold=int(fold))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)
        criterion = torch.nn.CrossEntropyLoss()

    elif dataset_name == "20ribs_landmark_classfication":
        num_classes = 20
        train_dataset = Landmark_Classfication("{}/{}/{}.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=False,fold=int(fold))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)


        test_dataset = Landmark_Classfication("{}/{}/{}.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=True,fold=int(fold))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)
        criterion = torch.nn.CrossEntropyLoss()

    elif dataset_name == "20_back_ribs_landmark_classfication":
        num_classes = 20
        train_dataset = Landmark_Classfication("{}/{}/{}.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=False,fold=int(fold))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)


        test_dataset = Landmark_Classfication("{}/{}/{}.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=True,fold=int(fold))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)
        criterion = torch.nn.CrossEntropyLoss()
    # elif dataset_name == "landmark_classfication_two_models":
    #     num_classes = 9
    #     train_dataset = Landmark_Classfication_Two_Models("/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_w_{}_mlp.npy".format(pretrain_mode, use_mlp),
    #                                                       "/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_w_{}_mlp.npy".format(disease_model, use_mlp), test=False,fold=int(fold), feature_merging=feature_mergeing)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)

    #     test_dataset = Landmark_Classfication_Two_Models("/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_w_{}_mlp.npy".format(pretrain_mode, use_mlp),
    #                                                       "/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_w_{}_mlp.npy".format(disease_model, use_mlp), test=True,fold=int(fold), feature_merging=feature_mergeing)
    #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)
    #     criterion = torch.nn.CrossEntropyLoss()
    elif dataset_name == "all_landmark_classfication":
        num_classes = 54
        train_dataset = Landmark_Classfication("{}/{}/{}.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=False,fold=int(fold))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)

        test_dataset = Landmark_Classfication("{}/{}/{}.npy".format(args.embd_dir, pretrain_mode, pretrain_mode), test=True,fold=int(fold))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)
        criterion = torch.nn.CrossEntropyLoss()
    # elif dataset_name == "all_landmark_classfication_two_models":
    #     num_classes = 54
    #     train_dataset = Landmark_Classfication_Two_Models("/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_all_landmarks_w_{}_mlp.npy".format( pretrain_mode, use_mlp),
    #                                                       "/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_all_landmarks_w_{}_mlp.npy".format(disease_model, use_mlp), test=False, fold=int(fold),feature_merging=feature_mergeing)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)

    #     test_dataset = Landmark_Classfication_Two_Models("/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_all_landmarks_w_{}_mlp.npy".format( pretrain_mode, use_mlp),
    #                                                       "/mnt/dfs/jpang12/datasets/nih_xray14/landmark_classfication_{}/data_dictionary_all_landmarks_w_{}_mlp.npy".format(disease_model, use_mlp), test=True, fold=int(fold),feature_merging=feature_mergeing)
    #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4,pin_memory=True, shuffle=True, drop_last=False)
    #     criterion = torch.nn.CrossEntropyLoss()



    if pretrain_mode in ['EVA-X', 'RAD-DINO']:
        mlp_in_feature = 768
    else:
        mlp_in_feature = 1024


    # mlp_in_feature = 2048 if feature_mergeing == "concat" else 1024


    mlp = MLP(num_classes= num_classes, mlp_in_feature = mlp_in_feature)
    optimizer = optim.AdamW(mlp.parameters(), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=5e-6)
    if torch.cuda.is_available():
        mlp = torch.nn.DataParallel(mlp, device_ids=[i for i in range(torch.cuda.device_count())])
        mlp = mlp.cuda()
        cudnn.benchmark = True

    best_test_performance = -100000

    patience = 0
    for epoch in range(0, 500):

        print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'], epoch))

        loss_avg = train_one_epoch(mlp ,criterion, optimizer, train_loader, epoch)
        print('Training loss: {}@Epoch: {}'.format(loss_avg, epoch))
        scheduler.step()

        # if epoch %5 ==0:
        accuracy = test_acc(mlp, test_loader)
        performance = accuracy
        print('Test accuracy: {}@Epoch: {}'.format(performance, epoch))
        patience += 1
        if patience > 100:
            print("Early stopping at epoch {}".format(epoch))
            break
        
        if performance > best_test_performance:
            patience = 0
            save_file = os.path.join(args.embd_dir, pretrain_mode, '{}_{}_fold{}.pth'.format(pretrain_mode, dataset_name, args.fold))
            save_model_wo_conf(mlp, optimizer, epoch+1, save_file)

            print( "Epoch {:04d}: test performance improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_test_performance, performance, save_file))
            best_test_performance = performance
            sys.stdout.flush()

    with open(os.path.join(args.embd_dir, pretrain_mode, '{}_{}_fold{}.txt'.format(pretrain_mode, dataset_name, args.fold)), 'w') as f:
        f.write('Best test performance: {}\n'.format(best_test_performance))
        f.write('Pretrain mode: {}\n'.format(pretrain_mode))
        f.write('Dataset name: {}\n'.format(dataset_name))
        f.write('Fold: {}\n'.format(fold))
        f.write('MLP in feature dimension: {}\n'.format(mlp_in_feature))
        f.write('Total epochs: {}\n'.format(epoch+1))
    print('Best accurracy: {}'.format(best_test_performance))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
    parser.add_argument('--dataset_name', type=str, default='9landmark_classfication',  help='9landmark_classfication, 20ribs_landmark_classfication, 20_back_ribs_landmark_classfication, landmark_classfication_two_models, all_landmark_classfication, all_landmark_classfication_two_models')
    parser.add_argument('--pretrain_mode', type=str, choices=['LeADER','adamv2','PEAC','ACE','Lamps','RAD-DINO','CheSS','EVA-X', 'ACEv2', 'ACEv2_swinv2', 'ark'], default='EVA-X', help="Choose the pretraining mode")
    parser.add_argument('--fold', type=int, default=1, help='Which fold to test')
    # parser.add_argument('--landmark_num', type=int, default=9,  help='the number of landmark points')
    parser.add_argument('--embd_dir', type=str, default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/landmark_classification_embedding',  help='key image embeddings saving dictionary.')
    parser.add_argument('--device', type=str, default='5',  help='device number')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    train_loops(args, dataset_name = args.dataset_name, pretrain_mode = args.pretrain_mode, fold=args.fold)


