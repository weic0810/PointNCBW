from __future__ import print_function
import argparse
import os
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.utils.data
import torch.nn.functional as F


from dataset_loader.modelnet import ModelNetDataLoader
from dataset_loader.shapenet_part import PartNormalDatasetLoader
from models.pointnet import PointNetCls, feature_transform_regularizer
from models.pointnet2_cls_ssg import PointNet2Cls


def set_random_seed(seed=2048):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pc_normalize_torch(pc):
    for i in range(len(pc)):
        centroid = torch.mean(pc[i].clone(), dim=0)
        pc[i] = pc[i].clone() - centroid
        m = torch.max(torch.sqrt(torch.sum(pc[i].clone().pow(2), dim=1)))
        pc[i] = pc[i].clone() / m
    return pc


def train(model, trainset, testset, optimizer, scheduler, epoch_num=200, batch_size=24, device='cuda', transpose=True, 
          feature_transform=False, seed=2048):

    if seed is None:
        manualSeed = random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
    else:
        manualSeed = seed
    set_random_seed(manualSeed)
    set_deterministic()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
   
    model.to(device)
    # train
    print('start training-------------------------------------')
    start_epoch = 0 
    
    for epoch in range(start_epoch, epoch_num):
        print("epoch: {}".format(epoch))
        # Training
        train_correct = 0
        total_trainset = 0
        total_loss = 0
        for i, (points, targets) in enumerate(trainloader):
            if transpose:
                points = points.transpose(2, 1)
            points, targets = points.to(device), targets.to(device)
            optimizer.zero_grad()
            model = model.train()
            pred, _, trans_feat = model(points)
            loss = F.nll_loss(pred, targets.long())
            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(targets).cpu().sum()
            train_correct += correct.item()
            total_trainset += points.size()[0]
        train_accuracy = train_correct / float(total_trainset)

        scheduler.step()

        # Test accuracy on test samples
        total_correct = 0
        total_testset = 0
        with torch.no_grad():
            for i, (points, targets) in enumerate(testloader):
                if transpose:
                    points = points.transpose(2, 1)
                points, targets = points.to(device), targets.to(device)
                model = model.eval()
                pred, _, trans_feat = model(points)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(targets).cpu().sum()
                total_correct += correct.item()
                total_testset += points.size()[0]
        test_accuracy = total_correct / float(total_testset)
        print("training epoch ", epoch, ", train ACC = ", train_accuracy, " , test ACC = ", test_accuracy, ", loss = ", total_loss)
        print("epoch {}: train ACC = {}, test ACC = {}, loss = {}".format(epoch, train_accuracy, test_accuracy, total_loss))
        
        torch.save(model.state_dict(), os.path.join(args.save_path, '{}.pth'.format(args.model)))
    
    return model


def get_model(model_name, num_classes):
   
    if model_name == 'pointnet':
        model = PointNetCls(k=num_classes)
    elif model_name == 'pointnet2':
        model = PointNet2Cls(num_class=num_classes)
    else:
        raise NotImplementedError("The model {} has not been implemented.".format(model_name))
    
    return model



def prepare_settings(args):

    if args.dataset == 'modelnet40':
        num_classes = 40
        trainset = ModelNetDataLoader(root=args.data_path, args=args, split='train')
        testset = ModelNetDataLoader(root=args.data_path, args=args, split='test')
    elif args.dataset == 'modelnet10':
        num_classes = 10
        trainset = ModelNetDataLoader(root=args.data_path, args=args, split='train')
        testset = ModelNetDataLoader(root=args.data_path, args=args, split='test')
    elif args.dataset == 'shapenet':
        num_classes = 16
        trainset = PartNormalDatasetLoader(root=args.data_path, split='train')
        testset = PartNormalDatasetLoader(root=args.data_path, split='test')
    else:
        raise NotImplementedError("The dataset {} has not been implemented now.".format(args.dataset))

    return trainset, testset, num_classes



def main(args):
    
    ## seed 2048
    set_random_seed()
    set_deterministic()
    
    trainset, testset, num_classes = prepare_settings(args)

    model = get_model(args.model, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.7)

    train(model, trainset, testset, optimizer, scheduler, args.epoch, args.batch_size, args.device, transpose=True)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--epoch', type=int, default=200, help='epoch in training')
    parser.add_argument('--model', default='pointnet', help='model name [default: pointnet]')
    parser.add_argument('--dataset', default='modelnet40', type=str, help='watermarked dataset')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save dataset offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--data_path', type=str, default='dataset/modelnet40_normal_resampled/',help="dataset path")
    parser.add_argument('--save_path', type=str, default='ckpt/surrogate/', help='path for saving models')
    parser.add_argument('--seed', type=int, default=2048, help='seed')

    args = parser.parse_args()
    main(args)
    
    
