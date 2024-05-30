from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *

class hourglass(nn.Module):
	def __init__(self, inplanes):
		super(hourglass, self).__init__()

		self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
								   nn.ReLU(inplace=True))

		self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

		self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
								   nn.ReLU(inplace=True))

		self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
								   nn.ReLU(inplace=True))

		self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
								   nn.BatchNorm3d(inplanes*2)) #+conv2

		self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
								   nn.BatchNorm3d(inplanes)) #+x

	def forward(self, x ,presqu, postsqu):
		
		out  = self.conv1(x) #in:1/4 out:1/8
		pre  = self.conv2(out) #in:1/8 out:1/8
		if postsqu is not None:
		   pre = F.relu(pre + postsqu, inplace=True)
		else:
		   pre = F.relu(pre, inplace=True)

		out  = self.conv3(pre) #in:1/8 out:1/16
		out  = self.conv4(out) #in:1/16 out:1/16

		if presqu is not None:
		   post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
		else:
		   post = F.relu(self.conv5(out)+pre, inplace=True) 

		out  = self.conv6(post)  #in:1/8 out:1/4

		return out, pre, post

class PSMNet(nn.Module):
	def __init__(self, maxdisp):
		super(PSMNet, self).__init__()
		self.maxdisp = maxdisp

		self.feature_extraction = feature_extraction()

		self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
									 nn.ReLU(inplace=True),
									 convbn_3d(32, 32, 3, 1, 1),
									 nn.ReLU(inplace=True))

		self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
								   nn.ReLU(inplace=True),
								   convbn_3d(32, 32, 3, 1, 1)) 

		self.dres2 = hourglass(32)

		self.dres3 = hourglass(32)

		self.dres4 = hourglass(32)

		self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
									  nn.ReLU(inplace=True),
									  nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

		self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
									  nn.ReLU(inplace=True),
									  nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

		self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
									  nn.ReLU(inplace=True),
									  nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()


	def forward(self, left, left_aug, right):

		refimg_fea	   = self.feature_extraction(left)
		targetimg_fea  = self.feature_extraction(right)
		G = CR_matrix(refimg_fea, targetimg_fea)

		if(left_aug is not None):
			self.training = True
		else:
			self.training = False

		if(self.training):
			augimg_fea	   = self.feature_extraction(left_aug)
			G_aug = CR_matrix(augimg_fea, targetimg_fea)
		else:
			G_aug = None


		#matching
		cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

		for i in range(self.maxdisp//4):
			if i > 0 :
			 cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
			 cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
			else:
			 cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
			 cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
		cost = cost.contiguous()

		cost0 = self.dres0(cost)
		cost0 = self.dres1(cost0) + cost0

		out1, pre1, post1 = self.dres2(cost0, None, None) 
		out1 = out1+cost0

		out2, pre2, post2 = self.dres3(out1, pre1, post1) 
		out2 = out2+cost0

		out3, pre3, post3 = self.dres4(out2, pre1, post2) 
		out3 = out3+cost0

		cost1 = self.classif1(out1)
		cost2 = self.classif2(out2) + cost1
		cost3 = self.classif3(out3) + cost2

		if self.training:
			cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
			cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

			cost1 = torch.squeeze(cost1,1)
			pred1 = F.softmax(cost1,dim=1)
			pred1 = disparityregression(self.maxdisp)(pred1)

			cost2 = torch.squeeze(cost2,1)
			pred2 = F.softmax(cost2,dim=1)
			pred2 = disparityregression(self.maxdisp)(pred2)

		cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
		cost3 = torch.squeeze(cost3,1)
		pred3 = F.softmax(cost3,dim=1)
		#For your information: This formulation 'softmax(c)' learned "similarity" 
		#while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
		#However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
		pred3 = disparityregression(self.maxdisp)(pred3)


		if self.training:
			cost_ = Variable(torch.FloatTensor(augimg_fea.size()[0], augimg_fea.size()[1]*2, self.maxdisp//4,  augimg_fea.size()[2],  augimg_fea.size()[3]).zero_()).cuda()
	
			for i in range(self.maxdisp//4):
				if i > 0 :
				 cost_[:, :augimg_fea.size()[1], i, :,i:]   = augimg_fea[:,:,:,i:]
				 cost_[:, augimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
				else:
				 cost_[:, :augimg_fea.size()[1], i, :,:]   = augimg_fea
				 cost_[:, augimg_fea.size()[1]:, i, :,:]   = targetimg_fea
			cost_ = cost_.contiguous()
	
			cost_0 = self.dres0(cost_)
			cost_0 = self.dres1(cost_0) + cost_0
	
			out_1, pre_1, post_1 = self.dres2(cost_0, None, None) 
			out_1 = out_1+cost_0
	
			out_2, pre_2, post_2 = self.dres3(out_1, pre_1, post_1) 
			out_2 = out_2+cost_0
	
			out_3, pre_3, post_3 = self.dres4(out_2, pre_1, post_2) 
			out_3 = out_3+cost_0
	
			cost_1 = self.classif1(out_1)
			cost_2 = self.classif2(out_2) + cost_1
			cost_3 = self.classif3(out_3) + cost_2
	
			cost_1 = F.upsample(cost_1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
			cost_2 = F.upsample(cost_2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
	
			cost_1 = torch.squeeze(cost_1,1)
			pred_1 = F.softmax(cost_1,dim=1)
			pred_1 = disparityregression(self.maxdisp)(pred_1)
	
			cost_2 = torch.squeeze(cost_2,1)
			pred_2 = F.softmax(cost_2,dim=1)
			pred_2 = disparityregression(self.maxdisp)(pred_2)
	
			cost_3 = F.upsample(cost_3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
			cost_3 = torch.squeeze(cost_3,1)
			pred_3 = F.softmax(cost_3,dim=1)
			#For your information: This formulation 'softmax(c)' learned "similarity" 
			#while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
			#However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
			pred_3 = disparityregression(self.maxdisp)(pred_3)




		if self.training:
			return pred1, pred2, pred3, pred_1, pred_2, pred_3, G, G_aug
		else:
			return pred3



def CR_matrix(left_f, right_f):
	B, C, H, W = left_f.size()
	left = left_f.view(B, C, H * W)
	right = right_f.view(B, C, H * W)
	
	G = torch.bmm(left, right.permute(0,2,1)).div(H * W)

	return G
