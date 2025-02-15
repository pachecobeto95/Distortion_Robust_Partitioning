import config, ee_dnns, sys, utils
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd

def extracting_ee_inference_time_edge_device_layer_level(args, test_loader, model, device):

	n_exits = args.n_branches + 1	
	inf_time_layer_list, inf_time_branches_list, target_list = [], [], []
	
	model.eval()
	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# Obtain confs and predictions for each side branch.
			inf_time_layer, inf_time_branches = model.forwardExtractingInferenceTimeLayerLevelEdge(data)

			inf_time_layer_list.append(inf_time_layer), inf_time_branches_list.append(inf_time_branches)
			target_list.append(target.item())
			n_layers = len(inf_time_layer)

	inf_time_layer_list, inf_time_branches_list = np.array(inf_time_layer_list), np.array(inf_time_branches_list)

	result_dict = {"device": len(target_list)*[str(device)]}

	
	for j in range(n_layers):
		result_dict["proc_time_layers_%s"%(j+1)] = inf_time_layer_list[:, j]

	for i in range(n_exits):
		result_dict["proc_time_branches_%s"%(i+1)] = inf_time_branches_list[:, i]

	#Converts to a DataFrame Format.
	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df

def main(args):

	n_classes = 257

	device_str = 'cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu'

	device = torch.device(device_str)

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inference_data")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	inf_data_path = os.path.join(inf_data_dir_path, "proc_time_layer_level_ee_%s_%s_branches_%s_id_%s_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id, args.location))
	
	ee_model = ee_dnns.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, 
		args.dim, args.exit_type, device, args.distribution)

	dataset_path = os.path.join("datasets", "caltech256")

	_, _, test_loader = utils.load_caltech256(args, dataset_path, indices_path)


	df_inf_time = extracting_ee_inference_time_edge_device_layer_level(args, test_loader, ee_model, device)

	df_inf_time.to_csv(inf_data_path, mode='a', header=not os.path.exists(inf_data_path))
	


if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech-256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "alexnet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	parser.add_argument('--input_dim', type=int, default=330, help='Input Dim.')

	parser.add_argument('--dim', type=int, default=300, help='Image dimension')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--use_gpu', type=bool, default=config.use_gpu, help='Use GPU? Default: %s'%(config.use_gpu))

	parser.add_argument('--n_branches', type=int, default=1, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution of the early exits. Default: %s'%(config.distribution))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	parser.add_argument('--model_id', type=int, default=3, help='Model_id.')

	parser.add_argument('--loss_weights_type', type=str, default="decrescent", help='loss_weights_type.')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	parser.add_argument('--location', type=str, help='Which machine extracts the inference data', choices=["pechincha", "jetson", "RO", "laptop"],
		default="RO")

	args = parser.parse_args()

	main(args)