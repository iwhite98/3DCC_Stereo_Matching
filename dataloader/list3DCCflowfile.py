import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

	classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
	image = [img for img in classes if img.find('frames_cleanpass') > -1]

	monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]


	monkaa_dir	= os.listdir(monkaa_path)

	all_left_img=[]
	test_left_img=[]


	for dd in monkaa_dir:
		for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
			if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
				all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)


	flying_path = filepath + [x for x in image if x == 'frames_cleanpass'][0]
	flying_dir = flying_path+'/TRAIN/'
	subdir = ['A','B','C']

	for ss in subdir:
		flying = os.listdir(flying_dir+ss)

		for ff in flying:
			imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
			for im in imm_l:
				if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
					all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)


	flying_dir = flying_path+'/TEST/'

	subdir = ['A','B','C']

	for ss in subdir:
		flying = os.listdir(flying_dir+ss)

		for ff in flying:
			imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
			for im in imm_l:
				if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
					test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)


	driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'

	subdir1 = ['35mm_focallength','15mm_focallength']
	subdir2 = ['scene_backwards','scene_forwards']
	subdir3 = ['fast','slow']

	for i in subdir1:
		for j in subdir2:
			for k in subdir3:
				imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')	  
				for im in imm_l:
					if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
						all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)



	return all_left_img, test_left_img

