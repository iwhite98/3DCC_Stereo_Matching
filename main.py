from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from dataloader import listflowfile as lt
from dataloader import list3DCCflowfile as ltc
from dataloader import SecenFlowLoader as DA

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                                        help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                                        help='select model')
parser.add_argument('--rgb_datapath', default='/media/jiaren/ImageNet/SceneFlowData/', help='rgb_datapath')
parser.add_argument('--aug_datapath', default='/media/jiaren/ImageNet/SceneFlowData/',help='aug_datapath')
parser.add_argument('--epochs', type=int, default=10,
                                        help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                                        help='load model')
parser.add_argument('--savemodel', default='./',
                                        help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                        help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
        torch.cuda.manual_seed(args.seed)

all_left_rgb, all_right_rgb, all_left_disp, test_left_rgb, test_right_rgb, test_left_disp, all_left_aug, test_left_aug = lt.dataloader(args.rgb_datapath, args.aug_datapath)
#all_left_aug, test_left_aug = ltc.dataloader(args.aug_datapath)

TrainImgLoader = torch.utils.data.DataLoader(
                 DA.myImageFloder(all_left_rgb, all_left_aug, all_right_rgb,all_left_disp, True), 
                 batch_size= 6, shuffle= True, num_workers= 8, drop_last=True)

TestImgLoader = torch.utils.data.DataLoader(
                 DA.myImageFloder(test_left_rgb, test_left_aug, test_right_rgb,test_left_disp, False), 
                 batch_size= 6, shuffle= False, num_workers= 4, drop_last=True)

print(len(all_left_rgb), len(all_left_aug))
print(len(test_left_rgb), len(test_left_aug))
print("num train : ", len(TrainImgLoader))
print("num test : ", len(TestImgLoader))


if args.model == 'stackhourglass':
        model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
        model = basic(args.maxdisp)
else:
        print('no model')

if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

if args.loadmodel is not None:
        print('Load pretrained model')
        pretrain_dict = torch.load(args.loadmodel)
        model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgL_aug, imgR, disp_L):
                model.train()

                if args.cuda:
                        imgL, imgL_aug, imgR, disp_true = imgL.cuda(), imgL_aug.cuda(), imgR.cuda(), disp_L.cuda()

           #---------
                mask = disp_true < args.maxdisp
                mask.detach_()
                #----
                optimizer.zero_grad()
                
                if args.model == 'stackhourglass':
                        output1, output2, output3, output1_aug, output2_aug, output3_aug, G, G_aug = model(imgL,imgL_aug, imgR)
                        output1 = torch.squeeze(output1,1)
                        output2 = torch.squeeze(output2,1)
                        output3 = torch.squeeze(output3,1)
                        output1_aug = torch.squeeze(output1_aug,1)
                        output2_aug = torch.squeeze(output2_aug,1)
                        output3_aug = torch.squeeze(output3_aug,1)
                        
                        loss_cr = F.smooth_l1_loss(G, G_aug, reduction = 'mean')
                        loss_ip = F.smooth_l1_loss(output3[mask], output3_aug[mask], reduction = 'mean')
                        loss_rgb = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
                        loss_aug = 0.5*F.smooth_l1_loss(output1_aug[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2_aug[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3_aug[mask], disp_true[mask], size_average=True) 
                        loss = loss_cr + loss_ip + loss_rgb + loss_aug

                elif args.model == 'basic':
                        output = model(imgL,imgR)
                        output = torch.squeeze(output,1)
                        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

                loss.backward()
                optimizer.step()

                return [loss_cr.data.cpu().detach(), loss_ip.data.cpu().detach(), loss_rgb.data.cpu().detach(), loss_aug.data.cpu().detach()]

def test(imgL,imgL_aug, imgR,disp_true):

                model.eval()
  
                if args.cuda:
                        imgL, imgL_aug, imgR, disp_true = imgL.cuda(), imgL_aug.cuda(), imgR.cuda(), disp_true.cuda()
                #---------
                mask = disp_true < 192
                #----

                if imgL.shape[2] % 16 != 0:
                        times = imgL.shape[2]//16               
                        top_pad = (times+1)*16 -imgL.shape[2]
                else:
                        top_pad = 0

                if imgL.shape[3] % 16 != 0:
                        times = imgL.shape[3]//16                                               
                        right_pad = (times+1)*16-imgL.shape[3]
                else:
                        right_pad = 0  

                imgL = F.pad(imgL,(0,right_pad, top_pad,0))
                imgL_aug = F.pad(imgL_aug,(0,right_pad, top_pad,0))
                imgR = F.pad(imgR,(0,right_pad, top_pad,0))

                with torch.no_grad():
                        output3 = model(imgL,None, imgR)
                        output3_aug = model(imgL_aug, None, imgR)
                        output3 = torch.squeeze(output3)
                        output3_aug = torch.squeeze(output3_aug)
                
                if top_pad !=0:
                        img = output3[:,top_pad:,:]
                        img_aug = output3_aug[:,top_pad:,:]
                else:
                        img = output3
                        img_aug = output3_aug

                if len(disp_true[mask])==0:
                   loss = 0
                else:
                   loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error
                   loss_aug = F.l1_loss(img_aug[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

                return [loss.data.cpu(), loss_aug.data.cpu()]

def adjust_learning_rate(optimizer, epoch):
        lr = 0.001
        print(lr)
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def main():
        start_full_time = time.time()
        max_loss = 100
        for epoch in range(0, args.epochs):
                print('This is %d-th epoch' %(epoch))
                total_train_loss = np.array([0.,0.,0.,0.])
                adjust_learning_rate(optimizer,epoch)

                ## training ##
                for batch_idx, (imgL_crop, imgL_aug, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
                        start_time = time.time()

                        loss_list = train(imgL_crop,imgL_aug, imgR_crop, disp_crop_L)
                        print("batch_idx : ", batch_idx, ", loss : ", np.round(loss_list,3))
                        total_train_loss = total_train_loss + loss_list
                print("epoch : ", epoch, ", total loss: ", np.round(total_train_loss,3)/len(TrainImgLoader))


        print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
        #------------- TEST ------------------------------------------------------------
        total_test_loss = np.array([0,0])
        for batch_idx, (imgL, imgL_aug, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss = test(imgL,imgL_aug, imgR, disp_L)
                print("batch_idx : ", batch_idx, ", loss : ", np.round(test_loss,3))
                total_test_loss = total_test_loss + test_loss

        print('total test loss = ', np.round(total_test_loss, 3)/len(TestImgLoader))
        cur_loss = total_test_loss[0]/len(TestImgLoader)
        if cur_loss < max_loss:
            max_loss = cur_loss
            print('save best epoch : ', epoch, max_loss)
            #SAVE
            savefilename = args.savemodel+'/best_model_dof'+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                                'train_loss': total_train_loss/len(TrainImgLoader),
            }, savefilename)


if __name__ == '__main__':
        main()
        
