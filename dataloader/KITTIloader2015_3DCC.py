import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import natsort

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(root_rgb, root_3dcc):

	left_rgb_fold  = 'image_2/'
	right_rgb_fold = 'image_3/'
	left_fog_fold = 'Left/fog_3d/'
	right_fog_fold = 'Right/fog_3d/'
	left_focus_fold = 'Left/near_focus/'
	right_focus_fold = 'Right/near_focus/'
	disp_L = 'disp_occ_0/'
	disp_R = 'disp_occ_1/'

	#intensity = ['1/', '2/', '3/', '4/', '5/']
	intensity =['1/']
	image = [img for img in os.listdir(root_rgb+left_rgb_fold) if img.find('_10') > -1]
	image = natsort.natsorted(image)

	train = image[:160]
	val   = image[160:]

	left_train = []
	left_train_aug = []
	right_train = []
	disp_train_L = []

	left_val = []
	left_val_aug = []
	right_val = []
	disp_val_L = []

	'''
	for img in train:
		left_train.append(root_rgb + left_rgb_fold + img)
		right_train.append(root_rgb + right_rgb_fold + img)
		left_train.append(root_rgb + disp_L + img)
	'''

	for idx in intensity:
		for img in train:
			## original image
			left_train.append(root_rgb + left_rgb_fold + img)
			right_train.append(root_rgb + right_rgb_fold + img)
			disp_train_L.append(root_rgb + disp_L + img)
			left_train_aug.append(root_3dcc + left_fog_fold + idx + img)
	
		'''
		for img in train:
			## original image
			left_train.append(root_rgb + left_rgb_fold + img)
			right_train.append(root_rgb + right_rgb_fold + img)
			disp_train_L.append(root_rgb + disp_L + img)

			## near focus image
			left_train.append(root_3dcc + left_focus_fold + idx + img)
			right_train.append(root_rgb + right_rgb_fold + img)
			disp_train_L.append(root_rgb + disp_L + img)
		'''

		for img in val:
			## original image
			left_val.append(root_rgb + left_rgb_fold + img)
			right_val.append(root_rgb + right_rgb_fold + img)
			disp_val_L.append(root_rgb + disp_L + img)
			left_val_aug.append(root_3dcc + left_fog_fold + idx + img)
		
		'''
		for img in val:
			## original image
			left_val.append(root_rgb + left_rgb_fold + img)
			right_val.append(root_rgb + right_rgb_fold + img)
			disp_val_L.append(root_rgb + disp_L + img)

			## near focus image
			left_val_aug.append(root_3dcc + left_focus_fold + idx + img)
			right_val_aug.append(root_rgb + right_rgb_fold + img)
			disp_val_L_aug.append(root_rgb + disp_L + img)
			'''

	return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L, left_train_aug, left_val_aug
