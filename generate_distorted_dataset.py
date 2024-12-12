import os, sys, cv2, argparse
import numpy as np
from tqdm import tqdm
import config 

class ImageProcessor(object):
	def __init__(self, distortion_type, distortion_lvl):
		self.distortion_type = distortion_type
		self.distortion_lvl = distortion_lvl

	def blur(self):
		self.dist_img = cv2.GaussianBlur(self.image, (self.distortion_lvl+1, self.distortion_lvl+1), 
			self.distortion_lvl)

	def noise(self):
		noise = np.random.normal(0, self.distortion_lvl, self.image.shape).astype(np.uint8)
		self.dist_img = cv2.add(self.image, noise)

	def distortion_not_found():
		raise ValueError("Invalid distortion type. Please choose 'blur' or 'noise'.")

	def apply(self, image_path):

		try:
			self.image = cv2.imread(image_path)
			dist_name = getattr(self, self.distortion_type, self.distortion_not_found)
			dist_name()

		except AttributeError as e:
			# Handle the exception
			print("An AttributeError occurred:", e)
			pass

	def save_distorted_image(self, output_path):
		cv2.imwrite(output_path, self.dist_img)


def generate_distorted_dataset(dataset_path, dist_type, dist_lvl, distorted_path):
	processor = ImageProcessor(dist_type, dist_lvl)

	for class_name in tqdm(os.listdir(dataset_path)):
		dir_class_path = os.path.join(dataset_path, class_name)
		dist_class_path = os.path.join(distorted_path, class_name)		
		os.makedirs(dist_class_path, exist_ok=True)		

		for filename in os.listdir(dir_class_path):
			imgPath = os.path.join(dir_class_path, filename)
			distorted_imgPath = os.path.join(dist_class_path, filename)
			if (os.path.isfile(imgPath)):
				try:
					processor.apply(imgPath)
					processor.save_distorted_image(distorted_imgPath)

				except cv2.error as e:
					print("error")
					pass


def main(args):

	distortion_levels = config.distortion_level_dict[args.distortion_type]

	for distortion_lvl in tqdm(distortion_levels):
		print("Distortion Level: %s"%(distortion_lvl))
		
		distorted_path = os.path.join(config.distorted_dataset_path, args.distortion_type, str(distortion_lvl))
		os.makedirs(distorted_path, exist_ok=True)
		generate_distorted_dataset(config.dataset_path, args.distortion_type, 
			distortion_lvl, distorted_path)



if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Generate distorted images.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default='caltech256', 
		choices=["caltech256"], help='Dataset name (default: Caltech-256)')

	parser.add_argument('--distortion_type', type=str, choices=["blur", "noise"], help='Distortion Type')

	args = parser.parse_args()

	main(args)