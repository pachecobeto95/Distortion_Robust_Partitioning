import config, ee_dnns, sys, utils
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import socket, pickle, time

def extracting_ee_outputs(args, test_loader, model, device):

	n_exits = args.n_branches + 1	

	model.eval()
	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# Obtain confs and predictions for each side branch.
			outputs = model.forwardExtractingOutputs(data)

			outputs = [data] + outputs

			break


	return outputs


import socket
import time
import pickle
import numpy as np

def extracting_ee_communication_time(outputs, n_rounds, SERVER_IP, SERVER_PORT):
    """
    Mede o tempo de comunicação para enviar as saídas de uma camada neural a um servidor remoto.

    Parâmetros:
        outputs (list): Lista de tensores de saída da camada neural.
        n_rounds (int): Número de rodadas de envio para calcular a média.
        SERVER_IP (str): Endereço IP do servidor remoto.
        SERVER_PORT (int): Porta do servidor remoto.

    Retorna:
        list: Lista de tempos médios de comunicação para cada saída.
    """
    avg_comm_times = {}  # Armazena os tempos médios de comunicação para cada saída

    for j, output in enumerate(outputs):
        send_time_list = []
        output_bytes = pickle.dumps(output)  # Serializa o tensor para bytes
        i = 1

        while (len(send_time_list) < n_rounds):
            client_socket = None  # Inicializa o socket como None

            try:
                # Cria um socket TCP
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5)  # Define um timeout de 5 segundos para a conexão

                # Conecta ao servidor remoto
                client_socket.connect((SERVER_IP, SERVER_PORT))

                # Mede o tempo de envio
                start_time = time.time()
                client_socket.sendall(output_bytes)  # Envia os dados
                end_time = time.time()

                # Calcula o tempo de envio
                send_time = end_time - start_time
                print(f"Rodada {i + 1}: Tempo de envio = {send_time:.4f} segundos")

                #if i > 2:  # Ignora as primeiras 5 rodadas para evitar warm-up
                send_time_list.append(send_time)
                print("Inserted: %s"%(len(send_time_list)))


                i+=1

            except ConnectionRefusedError:
                print(f"Erro: Conexão recusada. Verifique se o servidor está em execução em {SERVER_IP}:{SERVER_PORT}.")
            except socket.timeout:
                print(f"Erro: Timeout ao tentar conectar ao servidor {SERVER_IP}:{SERVER_PORT}.")
            except Exception as e:
                print(f"Erro inesperado: {e}")
            finally:
                # Fecha o socket, se estiver aberto
                if client_socket:
                    client_socket.close()

        # Calcula o tempo médio de comunicação para a saída atual
        if send_time_list:
            avg_comm_time = np.mean(send_time_list)
            
            if(j == 0):
                avg_comm_times["input"] = [avg_comm_time]
            else:
                avg_comm_times["comm_time_layers_%s"%(j)] = [avg_comm_time]
    
            print(f"CAMADA {j + 1}: Tempo médio de comunicação = {avg_comm_time:.4f} segundos")
        else:
        #    avg_comm_times.append(None)  # Adiciona None se não houver dados válidos
            print(f"CAMADA {j + 1}: Não foi possível calcular o tempo médio de comunicação.")

    return avg_comm_times


def main(args):

	n_classes = 257

	device_str = 'cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu'

	device = torch.device(device_str)

	SERVER_IP, SERVER_PORT = "51.158.1.21", 5200
	n_rounds = 10

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	comm_time_path = os.path.join(config.DIR_PATH, args.model_name, "inference_data", "comm_time_layer_level_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id))

	
	ee_model = ee_dnns.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, 
		args.dim, args.exit_type, device, args.distribution)

	dataset_path = os.path.join("undistorted_datasets", "caltech256")

	_, _, test_loader = utils.load_caltech256(args, dataset_path, indices_path)


	outputs = extracting_ee_outputs(args, test_loader, ee_model, device)

	avg_comm_times = extracting_ee_communication_time(outputs, n_rounds, SERVER_IP, SERVER_PORT)

	df = pd.DataFrame(avg_comm_times)
	df.to_csv(comm_time_path, index=False)




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

	args = parser.parse_args()

	main(args)