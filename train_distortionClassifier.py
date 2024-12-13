import config, utils, distortionClassifier
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd


def compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights):
	model_loss = 0
	ee_loss, acc_branches = [], []

	for i, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
		loss_branch = criterion(output, target)
		model_loss += weight*loss_branch

		acc_branch = 100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0)

		ee_loss.append(loss_branch.item()), acc_branches.append(acc_branch)

	acc_model = np.mean(np.array(acc_branches))

	return model_loss, ee_loss, acc_model, acc_branches


def trainDNN(model, train_loader, optimizer, criterion, epoch, device):

	loss_list, acc_list = [], []
	model.train()

	for i, (data, target) in enumerate(train_loader):
		print("Batch: %s/%s"%(i, len(train_loader)))
		data, target = data.to(device), target.to(device, dtype=torch.int64)
		optimizer.zero_grad()
		outputs = model(data)

		loss = criterion(outputs, target)
		loss.backward()
		optimizer.step()

		loss_list.append(loss.item())
		_, inf_label = torch.max(outputs, 1)
		acc = 100*(inf_label.eq(target.view_as(inf_label)).sum().item()/data.size(0))
		acc_list.append(acc)

	avg_acc, avg_loss = np.mean(acc_list), np.mean(loss_list)
	print("Epoch: %s, Avg Train Loss: %s, Avg Train Acc: %s"%(epoch, avg_loss, avg_acc))
	
	result = {"train_acc": avg_acc, "train_loss": avg_loss}

	return result
	


def evalDNN(model, val_loader, criterion, epoch, device):

	loss_list, acc_list = [], []
	model.eval()

	for i, (data, target) in enumerate(val_loader):
		data, target = data.to(device), target.to(device, dtype=torch.int64)
		outputs = model(data)
		loss = criterion(outputs, target)
		loss_list.append(loss.item())
		_, inf_label = torch.max(outputs, 1)
		acc = 100*(inf_label.eq(target.view_as(inf_label)).sum().item()/data.size(0))
		acc_list.append(acc)

	avg_acc, avg_loss = np.mean(acc_list), np.mean(loss_list)
	print("Epoch: %s, Avg Val Loss: %s, Avg Val Acc: %s"%(epoch, avg_loss, avg_acc))
	result = {"val_acc": avg_acc, "val_loss": avg_loss}

	return result



def main(args):

	models_dir_path = os.path.join(config.DIR_PATH, "DistortionClassifier", "models")
	history_dir_path = os.path.join(config.DIR_PATH, "DistortionClassifier", "history")

	os.makedirs(models_dir_path, exist_ok=True), os.makedirs(history_dir_path, exist_ok=True)

	model_save_path = os.path.join(models_dir_path, "distortionClassifier.pth")

	history_path = os.path.join(history_dir_path, "history_distortionClassifier.csv")

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	dataset_path = os.path.join(config.DIR_PATH, "distorted_datasets", "caltech256_fft")

	device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

	model = distortionClassifier.DistortionNet().to(device)

	lr = 0.01
	current_result = {}

	criterion = nn.CrossEntropyLoss()
	
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

	epoch, count_patience = 0, 0
	best_val_loss = np.inf
	df_history = pd.DataFrame()

	train_loader, val_loader, test_loader = utils.load_fft_caltech256(args, dataset_path, indices_path)

	while (count_patience < args.max_patience):
	#while (epoch < config.max_epochs):
		epoch += 1

		train_result = trainDNN(model, train_loader, optimizer, criterion, epoch, device)
		val_result = evalDNN(model, val_loader, criterion, epoch, device)

		current_result.update(train_result), current_result.update(val_result)
		df_history = pd.concat([df_history, pd.DataFrame([current_result])], ignore_index=True)
		df_history.to_csv(history_path)

		if (val_result["val_loss"] < best_val_loss):
			save_dict  = {}	
			best_val_loss = val_result["val_loss"]
			count_patience = 0

			save_dict.update(current_result)
			save_dict.update({"model_state_dict": model.state_dict(), "opt_state_dict": optimizer.state_dict()})
			torch.save(save_dict, model_save_path)

		else:
			count_patience += 1
			print("Current Patience: %s"%(count_patience))

	print("Stop! Patience is finished")


if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Training Early-exit DNN. These are the hyperparameters")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10"], help='Dataset name.')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	parser.add_argument('--input_dim', type=int, default=224, help='Input Dim.')

	parser.add_argument('--dim', type=int, default=224, help='Image dimension')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	#parser.add_argument('--cuda', type=bool, default=config.use_gpu, help='Use GPU? Default: %s'%(config.use_gpu))

	#parser.add_argument('--epochs', type=int, default=config.epochs, help='Epochs.')

	parser.add_argument('--max_patience', type=int, default=config.max_patience, help='Max Patience.')

	parser.add_argument('--model_id', type=int, help='Model_id.')

	parser.add_argument('--weight_decay', type=int, default=0.0005, help='Weight Decay.')

	args = parser.parse_args()

	main(args)