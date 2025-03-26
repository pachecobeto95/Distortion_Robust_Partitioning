import ee_dnns, sys, config
import argparse, logging, os, torch
import numpy as np
import torch.nn as nn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G):

    # Desenhar o grafo
    pos = nx.spring_layout(G)  # Definir o layout para posicionar os nós
    nx.draw(G, pos, with_labels=False, node_size=700, node_color='lightblue')

    # Adicionar rótulos nos nós
    nx.draw_networkx_labels(G, pos)

    # Exibir o grafo
    plt.show()


def build_edge_cloud_graph(ee_model, n_branches, weights):
    """
    Constrói o grafo do MobileNetV2 com 1 ramo lateral, adicionando pesos às arestas.

    Parâmetros:
        ee_model: Modelo com ramos laterais (early exits).
        n_branches (int): Número de ramos laterais.
        weights (dict): Dicionário contendo os pesos das arestas.

    Retorna:
        G (nx.DiGraph): Grafo direcionado com pesos nas arestas.
    """
    G = nx.DiGraph()

    i = 0
    G.add_node("input")
    G.add_node("output")

    for n_branch in range(n_branches + 1):
        for n_block in range(len(ee_model.stages[n_branch])):
            # Adicionar nós v[i]
            G.add_node("ve%s" % i)  # Nó de borda (edge)
            G.add_node("vc%s" % i)  # Nó de nuvem (cloud)

            # Adicionar arestas entre v[i-1] e v[i] (se i > 0)
            if i > 0:
                # Arestas entre nós de borda
                G.add_edge("ve%s" % (i - 1), "ve%s" % i, weight=weights[f"weight_edge_layer_{i}"])
                # Arestas entre nós de nuvem
                G.add_edge("vc%s" % (i - 1), "vc%s" % i, weight=weights[f"weight_cloud_layer_{i}"])
                # Aresta de borda para nuvem
                G.add_edge("ve%s" % (i - 1), "vc%s" % i, weight=weights[f"weight_comm_layer_{i}"])
            else:
                # Arestas do nó "input" para os primeiros nós de borda e nuvem
                G.add_edge("input", "ve%s" % i, weight=0)
                G.add_edge("input", "vc%s" % i, weight=weights["input"])

            i += 1

        # Adicionar o nó de bloco (b[n_branch])
        G.add_node("be%s" % n_branch)

        # Verificar se a aresta entre v[i-1] e v[i] existe antes de tentar removê-la
        if G.has_edge("ve%s" % (i - 1), "ve%s" % i):
            G.remove_edge("ve%s" % (i - 1), "ve%s" % i)

        # Criar as arestas com o nó de bloco
        G.add_edge("ve%s" % (i - 1), "be%s"%(n_branch), weight=weights[f"weight_edge_layer_{i}"])

        if n_branch < n_branches:
            G.add_edge("be%s" % n_branch, "ve%s" % i, weight=weights[f"weight_edge_branch_{n_branch+1}"])

        #i += 1  # Incrementar o índice i após a criação do nó de bloco

    # Adicionar arestas finais
    G.add_edge("be%s" % n_branches, "output", weight=weights[f"weight_edge_layer_{i}"])
    G.add_edge("vc21", "output", weight=0)  # Ajuste conforme necessário
    return G


def build_ee_dnn(ee_model, n_branches):

    G = nx.DiGraph()

    i = 0
    
    for n_branch in range(args.n_branches+1):
        for n_block in range(len(ee_model.stages[n_branch])):
            
            # Adicionar nó v[i]
            G.add_node("v%s" % i)

            # Adicionar aresta entre v[i-1] e v[i] (se i > 0)
            if i > 0:
                G.add_edge("v%s" % (i - 1), "v%s" % i)

            i += 1

        # Adicionar o nó de bloco (b[n_branch])
        G.add_node("b%s" % n_branch)

        # Verificar se a aresta entre v[i-1] e v[i] existe antes de tentar removê-la
        if G.has_edge("v%s" % (i - 1), "v%s" % (i)):
            G.remove_edge("v%s" % (i - 1), "v%s" % i)

        # Criar as arestas com o nó de bloco
        G.add_edge("v%s" % (i - 1), "b%s" % n_branch)
        
        if(n_branch < args.n_branches):
            G.add_edge("b%s" % n_branch, "v%s" % i)
        
        i += 1  # Incrementar o índice i após a criação do nó de bloco

    return G



def compute_prob_branches_math(df_inf_data, n_exit, threshold):
    """
    Calcula a probabilidade de classificação em cada ramo usando a expressão matemática:
    p_Y(k) = p_k * prod_{i=1}^{k-1} (1 - p_i)

    Parâmetros:
        df_inf_data (pd.DataFrame): DataFrame contendo as colunas 'conf_branch_1', 'conf_branch_2', etc.
        n_exit (int): Número de ramos (exits) a serem considerados.
        threshold (float): Limiar de confiança para classificação antecipada.

    Retorna:
        list: Lista de probabilidades de classificação em cada ramo.
    """
    prob_list = []
    remaining_df = df_inf_data  # Inicializa com o DataFrame completo
    prod_not_classified = 1.0  # Inicializa o produto das probabilidades de não classificação

    for i in range(n_exit):
        # Verifica se a coluna do ramo atual existe
        conf_column = f"conf_branch_{i+1}"
        if conf_column not in remaining_df.columns:
            raise ValueError(f"Coluna '{conf_column}' não encontrada no DataFrame.")

        # Calcula p_k: P[Conf_k >= threshold]
        ee_samples = remaining_df[conf_column] >= threshold
        df_ee = remaining_df[ee_samples]

        if len(remaining_df) == 0:
            p_k = 0.0  # Evita divisão por zero se não houver amostras restantes
        else:
            p_k = df_ee.shape[0] / len(remaining_df)

        # Calcula p_Y(k) = p_k * prod_{i=1}^{k-1} (1 - p_i)
        p_Y_k = prod_not_classified
        prob_list.append(p_Y_k)

        # Atualiza o produto das probabilidades de não classificação
        prod_not_classified *= (1 - p_k)

        # Atualiza o DataFrame para as amostras que não foram classificadas no ramo atual
        remaining_df = remaining_df[~ee_samples]

    return prob_list



def compute_weights(args, model, df_inf_data, df_proc_time_edge, df_proc_time_cloud, df_comm_time, threshold):

    n_layer = 0

    n_exit = args.n_branches + 1
    weights = {}

    prob_list = compute_prob_branches_math(df_inf_data, n_exit, threshold)

    for n_branch in range(n_exit):
        for n_block in range(len(model.stages[n_branch])):

            print(n_layer)


            if(n_layer+1 == 21):
                weights["weight_edge_layer_%s"%(n_layer+1)] = prob_list[n_branch]*df_proc_time_edge["proc_time_branches_%s"%(n_branch+1)].mean()
                weights["weight_cloud_layer_%s"%(n_layer+1)] = prob_list[n_branch]*df_proc_time_cloud["proc_time_branches_%s"%(n_branch+1)].mean()
                weights["weight_comm_layer_%s"%(n_layer+1)] = prob_list[n_branch]*df_comm_time["comm_time_layers_%s"%(n_layer+1)].mean()

            else:

                weights["weight_edge_layer_%s"%(n_layer+1)] = prob_list[n_branch]*df_proc_time_edge["proc_time_layers_%s"%(n_layer+1)].mean()
                weights["weight_cloud_layer_%s"%(n_layer+1)] = prob_list[n_branch]*df_proc_time_cloud["proc_time_layers_%s"%(n_layer+1)].mean()
                weights["weight_comm_layer_%s"%(n_layer+1)] = prob_list[n_branch]*df_comm_time["comm_time_layers_%s"%(n_layer+1)].mean()


            n_layer += 1


        weights["weight_edge_branch_%s"%(n_branch+1)] = prob_list[n_branch]*df_proc_time_edge["proc_time_branches_%s"%(n_branch+1)].mean()

        #n_layer += 1

    weights["input"] = df_comm_time["input"].mean()


    return weights






def main(args):

    n_classes = 257

    device_str = 'cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu'

    device = torch.device(device_str)

    inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inference_data")

    dist_inf_data_path = os.path.join(inf_data_dir_path, "%s_inf_data_ee_%s_%s_branches_%s_id_%s_%s.csv"%(args.distortion_type, args.model_name, 
        args.n_branches, args.loss_weights_type, args.model_id, args.location))

    proc_time_cloud_path = os.path.join(inf_data_dir_path, "proc_time_layer_level_ee_%s_%s_branches_%s_id_%s_desktop_RO.csv"%(args.model_name, 
        args.n_branches, args.loss_weights_type, args.model_id))

    proc_time_edge_path = os.path.join(inf_data_dir_path, "proc_time_layer_level_ee_%s_%s_branches_%s_id_%s_laptop.csv"%(args.model_name, 
        args.n_branches, args.loss_weights_type, args.model_id))


    comm_time_path = os.path.join(inf_data_dir_path, "comm_time_layer_level_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, 
        args.n_branches, args.loss_weights_type, args.model_id))

    df_inf_data, df_proc_time_edge, df_proc_time_cloud = pd.read_csv(dist_inf_data_path), pd.read_csv(proc_time_edge_path), pd.read_csv(proc_time_cloud_path)
    
    df_comm_time = pd.read_csv(comm_time_path)

    ee_model = ee_dnns.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, 
        args.dim, args.exit_type, device, args.distribution)

    threshold_list = [0.7, 0.8, 0.9]

    for distortion_lvl in config.distortion_level_dict[args.distortion_type]:
        print("Distortion Type: %s, Distortion Level: %s"%(args.distortion_type, distortion_lvl))

        df_filter_inf_data = df_inf_data[(df_inf_data.distortion_lvl==distortion_lvl) & (df_inf_data.distortion_type==args.distortion_type)]

        for threshold in threshold_list:

            weights = compute_weights(args, ee_model, df_filter_inf_data, df_proc_time_edge, df_proc_time_cloud, df_comm_time, threshold)

            ee_dnn_graph = build_edge_cloud_graph(ee_model, args.n_branches, weights)




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

    parser.add_argument('--location', type=str, help='Which machine extracts the inference data', choices=["pechincha", "jetson", "RO"],
        default="RO")

    parser.add_argument('--distortion_type', type=str, help='Distoriton Type applyed in dataset.',
        choices=["blur", "noise", "pristine"])


    args = parser.parse_args()

    main(args)