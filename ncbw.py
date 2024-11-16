from __future__ import print_function
import argparse
import os
import random
import time
import numpy as np
import torch.utils.data
import random
import scipy
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


from train import prepare_settings, train, get_model
from model.pointnet import PointNetCls, feature_transform_regularizer
from dataset_loader.modelnet import ModelNetDataLoader
from dataset_loader.shapenet_part import PartNormalDataset

from model.pointnet import PointNetCls



def pc_normalize_torch(pc):
    for i in range(len(pc)):
        centroid = torch.mean(pc[i].clone(), dim=0)
        pc[i] = pc[i].clone() - centroid
        m = torch.max(torch.sqrt(torch.sum(pc[i].clone().pow(2), dim=1)))
        pc[i] = pc[i].clone() / m
    return pc


def get_ind(trainset, target_cls, watermark_num, num_classes):
    others_ind = []
    target_ind = []
    class_start_ind = [-1] * (num_classes+1)
    for i in range(len(trainset)):
        if trainset[i][1] != target_cls:
            others_ind.append(i)
        else:
            target_ind.append(i)
        if class_start_ind[int(trainset[i][1])] == -1:
            class_start_ind[int(trainset[i][1])] = i
    class_start_ind[num_classes] = len(trainset)
    surrogate_ind = random.sample(target_ind, watermark_num)  
    return surrogate_ind, class_start_ind, target_ind


def get_trigger(wr_path, num_points):
    trigger = np.loadtxt(wr_path, delimiter=',').astype(np.float32)
    return trigger[:num_points]


def get_represents(trainset, model, inds, device):
    model = model.to(device)
    model.eval()
    represents = torch.empty(0)
    for id in inds:
        pts = torch.from_numpy(trainset[id][0]).to(device)
        with torch.no_grad():
            _, rep, _ = model(pts.unsqueeze(dim=0).transpose(2,1))
        represents = torch.cat((represents, rep.unsqueeze(dim=0).cpu()), dim=0)

    return represents



def optimize_shape(model, pts, target_represents, device):
    model.eval()

    pts = torch.from_numpy(pts)
    target_represents = target_represents.to(device)
    Ds = np.array([])
    dict = {}
    for i in range(30):
        alpha = (torch.rand(1)-0.5) * torch.pi * 2  #(-pi/6, pi/6)
        beta = (torch.rand(1)-0.5) * torch.pi * 2
        theta = (torch.rand(1)-0.5) * torch.pi * 2
        rotation_x, rotation_y, rotation_z = get_rotation(alpha, beta, theta)
        rotated_pc = torch.matmul(pts, torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z)))
        _, represent, _ = model(rotated_pc.unsqueeze(dim=0).transpose(2, 1).to(device))
        D = 0
        for target_rep in target_represents:
            D += (target_rep.detach() - represent.detach()).pow(2).sum().sqrt()
        
        Ds = np.append(Ds, D.cpu().numpy())
        dict[i] = [alpha, beta, theta]
    # select the optimal starting point
    index = np.argsort(Ds)[0]
    alpha, beta, theta = dict[index]

    opt = torch.optim.Adam([alpha, beta, theta], lr=0.025, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    for epoch in range(20):
        opt.zero_grad()
        rotation_x, rotation_y, rotation_z = get_rotation(alpha, beta, theta)
        rotation_x.requires_grad = True
        rotation_y.requires_grad = True
        rotation_z.requires_grad = True
        rotated_pc = torch.matmul(pts, torch.matmul(rotation_x, torch.matmul(rotation_y, rotation_z)))
        _, represent, _ = model(rotated_pc.unsqueeze(dim=0).transpose(2, 1).to(device))

        D = 0
        for target_rep in target_represents:
            D += (target_rep.detach() - represent).pow(2).sum().sqrt()
        D.backward()
        alpha.grad, beta.grad, theta.grad = get_angle_grad(alpha, beta, theta, rotation_x, rotation_y, rotation_z)
        opt.step()
        scheduler.step()
    # print('after rot, D = ', D.item())
   
    return rotated_pc.detach().cpu().numpy(), D.item()


def optimize_point(model, pc, target_represents, device,  step=20, decay=1, alpha=0.01, la=50):
    model.eval()
    pts = torch.from_numpy(pc).to(device)
    target_represents = target_represents.to(device)
    momentum = torch.zeros_like(pts).to(device)
    adv_pts = pts.clone()
    for i in range(step):
        adv_pts.requires_grad = True
        _, rep, _ = model(adv_pts.unsqueeze(0).transpose(2,1))
        cost = 0
        for target_rep in target_represents:
            cost = cost + (target_rep.detach() - rep).pow(2).sum().sqrt()
        cost = cost/len(target_rep) + la * (pts.detach() - adv_pts).pow(2).sum()
        grad = torch.autograd.grad(cost, adv_pts,
                                       retain_graph=False, create_graph=False)[0]
        grad = grad / grad.pow(2).sum().sqrt()
        # gather momentum
        grad = grad + momentum*decay
        momentum = grad
        if i % 10 == 0:
            alpha = alpha / 10
        adv_pts = adv_pts.detach() - alpha*grad.sign()
    
    return adv_pts.detach().cpu().numpy(), cost.item()


def get_rotation(alpha, beta, theta):
    rotations_x = torch.tensor(
        [[1, 0, 0], [0, torch.cos(alpha), -torch.sin(alpha)],
         [0, torch.sin(alpha), torch.cos(alpha)]]
    )
    rotation_y = torch.tensor(
        [[torch.cos(beta), 0, torch.sin(beta)], [0, 1, 0],
         [-torch.sin(beta), 0, torch.cos(beta)]]
    )
    rotation_z = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta), 0],
         [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]]
    )
    return rotations_x, rotation_y, rotation_z


def get_angle_grad(alpha, beta, theta, rotation_x, rotation_y, rotation_z):
    alpha_grad = ((rotation_x.grad[1][1].detach())) * (-torch.sin(alpha.data)) + \
                 ((rotation_x.grad[1][2].detach())) * (-torch.cos(alpha.data)) + \
                 ((rotation_x.grad[2][1].detach())) * (torch.cos(alpha.data)) + \
                 ((rotation_x.grad[2][2].detach())) * (-torch.sin(alpha.data))
    beta_grad =  ((rotation_y.grad[0][0].detach())) * (-torch.sin(beta.data)) + \
                 ((rotation_y.grad[0][2].detach())) * (torch.cos(beta.data)) + \
                 ((rotation_y.grad[2][0].detach())) * (-torch.cos(beta.data)) + \
                 ((rotation_y.grad[2][2].detach())) * (-torch.sin(beta.data))
    theta_grad = ((rotation_z.grad[0][0].detach())) * (-torch.sin(theta.data)) + \
                 ((rotation_z.grad[0][1].detach())) * (-torch.cos(theta.data)) + \
                 ((rotation_z.grad[1][0].detach())) * (torch.cos(theta.data)) + \
                 ((rotation_z.grad[1][1].detach())) * (-torch.sin(theta.data))
    return alpha_grad, beta_grad, theta_grad


def test_pvalue(tao, model, certify_set, trigger, insert_ind, transpose, device):
    model.eval()
    watermark = watermark.to(device)
    pred_pro_reduce = np.array([])
    pred_diff_num = 0 
    for (pts, label) in certify_set:
        # label = torch.from_numpy(pts).to(device)
        pts = torch.from_numpy(pts).to(device)
        watermark_pts = pts.clone()
        watermark_pts[insert_ind] = trigger
        if transpose:
            clean_pred, _, _ = model(pts.unsqueeze(dim=0).transpose(2, 1))
            pred, _, _ = model(watermark_pts.unsqueeze(dim=0).transpose(2, 1))
        else:
            clean_pred, _, _ = model(pts.unsqueeze(dim=0))
            pred, _, _ = model(watermark_pts.unsqueeze(dim=0))
        clean_pred_choice = clean_pred.data.max(1)[1]
        pred_choice = pred.data.max(1)[1]
        clean_pred_choice = clean_pred.data.max(1)[1]
        if clean_pred_choice != pred_choice:
            pred_diff_num += 1
        if clean_pred_choice == label: # only on rightly classified samples
            clean_pred_source_pro = clean_pred.data[0,clean_pred_choice].exp()
            pred_source_pro = pred.data[0,clean_pred_choice].exp()
            pred_pro_reduce = np.append(pred_pro_reduce, (clean_pred_source_pro - pred_source_pro).cpu().numpy()) 
    # print(pred_pro_reduce)
    _, pvalue = scipy.stats.ttest_1samp(pred_pro_reduce, popmean=tao, alternative='greater')
    print("pvalue = ", pvalue)
    return pvalue, pred_diff_num/len(certify_set)



def optimize_watermark(args):

    target_cls = args.target
    target_num = args.tn
    watermark_num = args.wn
    device = args.device
    
    trainset, testset, num_classes = prepare_settings(args)

    model = get_model(args.model, num_classes)

    ckpt_dir = './pretrain/surrogate/{}.pth'.format(args.model)
    if os.path.exists(ckpt_dir):
        model.load_state_dict(torch.load(ckpt_dir, map_location=torch.device('cpu')))
        print("pretrained model loaded")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.7)

        model = train(model=model, trainset=trainset, testset=testset, optimizer=optimizer, scheduler=scheduler, device=args.device)
        
        torch.save(model.state_dict(), os.path.join(args.surrogate_path , f'{args.model}_{args.dataset}.pth'))

    model = model.to(args.device)
    model.eval()

    # prepare trigger
    trigger = get_trigger(args.watermark_path, args.num_points)
    
    # prepare samples for watermark
    surrogate_ind, class_start_index, target_ind = get_ind(trainset, target_cls, watermark_num, num_classes)

    # get the representations of target class through surrogate samples
    target_represents = get_represents(trainset, model, surrogate_ind, device)

    watermark_indexes = np.array([])
    watermark_distances = np.array([])
    index_pts_dict = {}
  
    for i in range(len(class_start_index)-1):
        if i == target_ind:
            continue
        # Apply watermarks evenly across all classes (except target class)
        # Number of watermarked samples is proportional to class size
        class_watermark_num = int(max((watermark_num/len(trainset))*(class_start_index[i+1]-class_start_index[i]),1))
        represent_distances = np.array([])
        potential_indexes = np.array([])
        for index in range(class_start_index[i], class_start_index[i+1]):
            start_time = time.time()
            potential_indexes = np.append(potential_indexes, index)
            pts = trainset[index][0].copy()
            rotated_pts, _ = optimize_shape(model, pts, target_represents, device) 
            new_pts, distance = optimize_point(model, rotated_pts, target_represents, device, la=args.la)
            print('D = ', distance)
            # print('Dc = ', chamfer_distance(trainset[index][0], new_pts))
            represent_distances = np.append(represent_distances, distance)
            index_pts_dict[index] = [new_pts, rotated_pts, trainset[index][0]]
            end_time = time.time()
            print('time costs ', end_time-start_time)
            print('-------------------')

        watermark_choice = np.argsort(represent_distances)[:class_watermark_num] # select the optimal candidate
        watermark_indexes = np.concatenate((watermark_indexes, potential_indexes[watermark_choice]), axis=0)
        watermark_distances = np.concatenate((watermark_distances, represent_distances[watermark_choice]))

    # print('watermark inds are')
    # print(watermark_indexes)

    insert_ind = np.random.choice(range(args.num_points), int(args.num_points*0.05), replace=False) 

    watermark_trainset = []
    for i in range(len(trainset)):
        if i in watermark_indexes:
            pts = index_pts_dict[i][0]
            pts[insert_ind] = trigger
            watermark_trainset.append((pts, trainset[i][1]))
        else:
            watermark_trainset.append(trainset[i])
        # here you can save the watermarked dataset 
        # np.savetxt(path_to_your_watemarked_dataset, watermark_trainset[i][0])

    model_names = ['pointnet']  # model to be trained on watermarked dataset

    source_indexes = np.asarray(source_indexes)
    watermark_indexes = watermark_indexes.astype(np.int64)
    source_indexes = source_indexes.astype(np.int64)
    watermark_indexes = watermark_indexes.astype(np.int64)

    certify_set = [trainset[i] for i in target_ind]

    for model_name in model_names:
       
        model = get_model(model_name, num_classes)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
       
        model = train(model=model, trainset=watermark_trainset, testset=testset, optimizer=optimizer, scheduler=scheduler)
        
        
        torch.save(model.state_dict(), os.path.join('pretrain', 'watermark' , f'watermarked_{args.source}_{args.model}_{model_name}_{args.dataset}.pth'))
        # torch.save(model.state_dict(), os.path.join('pretrain', 'watermark' , 'bw6_final', args.model, 's'+str(SOURCE)+'_wn'+str(WATERMARK_NUM)+'_tn'+str(TRIGGER_NUM)+'_Dc'+str(Dc)+'-'+model_name+'_'+args.dataset+'.pth'))
        

        print('test  model ', model_name)

        trigger = get_trigger(args.watermark_path, args.num_points)
        trigger_torch = torch.from_numpy(trigger)

        pvalue, wsr = test_pvalue(tao=0.2, model=model, certify_set=certify_set, trigger=trigger_torch, 
                             insert_ind=insert_ind, device=device)
       
        print('pvalue is {}, wsr is {}'.format(pvalue, wsr))

    return     


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save dataset offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--device', type=str, default='cuda:0', help='specify gpu device')
    parser.add_argument('--dataset', type=str, default='modelnet40', help="the name of dataset")
    parser.add_argument('--data_path', type=str, default='dataset/modelnet40_normal_resampled/',help="path of dataset")
    parser.add_argument('--model', type=str, default="pointnet", help='source model')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--tn', type=int, default=10, help='num of sample in target class for surrogate representations')
    parser.add_argument('--wn', type=int, default=120, help='num of watermarked samples')
    # parser.add_argument('--wr', type=float, default=0.1, help='watermark rate of one class')
    parser.add_argument('--step', type=int, default=20, help='step in optimization')
    parser.add_argument('--la', type=float, default=50, help='lambda of shape-wise optimization')
    parser.add_argument('--target', type=int, default=18, help='class of selected samples')
    parser.add_argument('--trigger_path', type=str, default='trigger/sphere.txt', help='path of trigger pattern') 
    parser.add_argument('--surrogate_path', type=str, default='ckpt/surrogate/', help='path of surrogate models') 
    parser.add_argument('--watermark_path', type=str, default='ckpt/watermark/', help='path of watermarked models') 
    parser.add_argument('--seed', type=int, default=2048, help='seed')
    args = parser.parse_args()

    if args.seed is None:
        manualSeed = random.randint(1, 10000)
        print("Random Seed: ", manualSeed)
    else:
        manualSeed = args.seed
        print("Seed: ", args.seed)

    def set_random_seed(seed=2048):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    set_random_seed(manualSeed)
    
    optimize_watermark(args)
    
    

