import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageEnhance

import natsort
from torch.utils.data import Dataset

class Augmentor:
	def __init__(
		self,
		image_height=384,
		image_width=512,
		max_disp=256,
		scale_min=0.6,
		scale_max=1.0,
		seed=0,
	):
		super().__init__()
		self.image_height = image_height
		self.image_width = image_width
		self.max_disp = max_disp
		self.scale_min = scale_min
		self.scale_max = scale_max
		self.rng = np.random.RandomState(seed)

	def chromatic_augmentation(self, img):
		random_brightness = np.random.uniform(0.8, 1.2)
		random_contrast = np.random.uniform(0.8, 1.2)
		random_gamma = np.random.uniform(0.8, 1.2)

		img = Image.fromarray(img)

		enhancer = ImageEnhance.Brightness(img)
		img = enhancer.enhance(random_brightness)
		enhancer = ImageEnhance.Contrast(img)
		img = enhancer.enhance(random_contrast)

		gamma_map = [
			255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
		] * 3
		img = img.point(gamma_map)	# use PIL's point-function to accelerate this part

		img_ = np.array(img)

		return img_

	def __call__(self, left_img, right_img, left_disp):
		# 1. chromatic augmentation
		left_img = self.chromatic_augmentation(left_img)
		right_img = self.chromatic_augmentation(right_img)

		# 2. spatial augmentation
		# 2.1) rotate & vertical shift for right image
		if self.rng.binomial(1, 0.5):
			angle, pixel = 0.1, 2
			px = self.rng.uniform(-pixel, pixel)
			ag = self.rng.uniform(-angle, angle)
			image_center = (
				self.rng.uniform(0, right_img.shape[0]),
				self.rng.uniform(0, right_img.shape[1]),
			)
			rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
			right_img = cv2.warpAffine(
				right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
			)
			trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
			right_img = cv2.warpAffine(
				right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
			)

		# 2.2) random resize
		resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

		left_img = cv2.resize(
			left_img,
			None,
			fx=resize_scale,
			fy=resize_scale,
			interpolation=cv2.INTER_LINEAR,
		)
		right_img = cv2.resize(
			right_img,
			None,
			fx=resize_scale,
			fy=resize_scale,
			interpolation=cv2.INTER_LINEAR,
		)

		disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
		disp_mask = disp_mask.astype("float32")
		disp_mask = cv2.resize(
			disp_mask,
			None,
			fx=resize_scale,
			fy=resize_scale,
			interpolation=cv2.INTER_LINEAR,
		)

		left_disp = (
			cv2.resize(
				left_disp,
				None,
				fx=resize_scale,
				fy=resize_scale,
				interpolation=cv2.INTER_LINEAR,
			)
			* resize_scale
		)

		# 2.3) random crop
		h, w, c = left_img.shape
		dx = w - self.image_width
		dy = h - self.image_height
		dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
		dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

		M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
		left_img = cv2.warpAffine(
			left_img,
			M,
			(self.image_width, self.image_height),
			flags=cv2.INTER_LINEAR,
			borderValue=0,
		)
		right_img = cv2.warpAffine(
			right_img,
			M,
			(self.image_width, self.image_height),
			flags=cv2.INTER_LINEAR,
			borderValue=0,
		)
		left_disp = cv2.warpAffine(
			left_disp,
			M,
			(self.image_width, self.image_height),
			flags=cv2.INTER_LINEAR,
			borderValue=0,
		)
		disp_mask = cv2.warpAffine(
			disp_mask,
			M,
			(self.image_width, self.image_height),
			flags=cv2.INTER_LINEAR,
			borderValue=0,
		)

		# 3. add random occlusion to right image
		if self.rng.binomial(1, 0.5):
			sx = int(self.rng.uniform(50, 100))
			sy = int(self.rng.uniform(50, 100))
			cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
			cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
			right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
				np.mean(right_img, 0), 0
			)[np.newaxis, np.newaxis]

		return left_img, right_img, left_disp, disp_mask


class KITTI20153DCCDataset(Dataset):
	def __init__(self, root_rgb, root_3dcc):
		super().__init__()

		left_rgb_fold  = 'image_2/'
		right_rgb_fold = 'image_3/'
		left_fog_fold = 'Left/fog_3d/'
		right_fog_fold = 'Right/fog_3d/'
		left_focus_fold = 'Left/near_focus/'
		right_focus_fold = 'Right/near_focus/'
		disp_L = 'disp_occ_0/'
		disp_R = 'disp_occ_1/'

		intensity = ['1/', '2/', '3/', '4/', '5/']
		image = [img for img in os.listdir(root_rgb + left_rgb_fold) if img.find('_10') > -1]
		image = natsort.natsorted(image)
		image = image[:180]
		#image = ["000000_10.png"]

		self.total_left_imgs = []
		self.total_right_imgs = []
		self.total_left_disp = []
		self.total_right_disp = []

		for img in image:
			self.total_left_imgs.append(root_rgb + left_rgb_fold + img)
			self.total_right_imgs.append(root_rgb + right_rgb_fold + img)
			self.total_left_disp.append(root_rgb + disp_L + img)
			self.total_right_disp.append(root_rgb + disp_R + img)

		for idx in intensity:
			for img in image:
				self.total_left_imgs.append(root_3dcc + left_fog_fold + idx + img)
				self.total_right_imgs.append(root_rgb + right_rgb_fold + img)
				self.total_left_disp.append(root_rgb + disp_L + img)
				self.total_right_disp.append(root_rgb + disp_R + img)
	
			for img in image:
				self.total_left_imgs.append(root_3dcc + left_focus_fold + idx + img)
				self.total_right_imgs.append(root_rgb + right_rgb_fold + img)
				self.total_left_disp.append(root_rgb + disp_L + img)
				self.total_right_disp.append(root_rgb + disp_R + img)

		self.augmentor = Augmentor(
			image_height=384,
			image_width=512,
			max_disp=256,
			scale_min=0.6,
			scale_max=1.0,
			seed=0,
		)
		self.rng = np.random.RandomState(0)

	def get_disp(self, path):
		disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
		return disp.astype(np.float32) / 512

	def __getitem__(self, index):
		# find path
		left_path = self.total_left_imgs[index]
		right_path = self.total_right_imgs[index]
		left_disp_path = self.total_left_disp[index]
		right_disp_path = self.total_right_disp[index]

		# read img, disp
		left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
		right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
		left_disp = self.get_disp(left_disp_path)
		right_disp = self.get_disp(right_disp_path)

		left_disp[left_disp == np.inf] = 0

		# augmentaion
		left_img, right_img, left_disp, disp_mask = self.augmentor(
			left_img, right_img, left_disp
		)

		left_img = left_img.transpose(2, 0, 1).astype("uint8")
		right_img = right_img.transpose(2, 0, 1).astype("uint8")

		return {
			"left": left_img,
			"right": right_img,
			"disparity": left_disp,
			"mask": disp_mask,
		}

	def __len__(self):
		return len(self.total_left_imgs)
