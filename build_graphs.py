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

def build_edge_cloud_graph(ee_model, n_branches):

    G = nx.DiGraph()

    i = 0

    G.add_node("input")
    G.add_node("output")

    for n_branch in range(args.n_branches+1):
        for n_block in range(len(ee_model.stages[n_branch])):

            # Adicionar nó v[i]
            G.add_node("ve%s" % i)
            G.add_node("vc%s" % i)

            # Adicionar aresta entre v[i-1] e v[i] (se i > 0)
            if i > 0:
                G.add_edge("ve%s" % (i - 1), "ve%s" % i)
                G.add_edge("vc%s" % (i - 1), "vc%s" % i)
                G.add_edge("ve%s" % (i - 1), "vc%s" % i)

            else:
                G.add_edge("input", "ve%s" % i)
                G.add_edge("input", "vc%s" % i)
            i+=1


        # Adicionar o nó de bloco (b[n_branch])
        G.add_node("be%s" % n_branch)

        # Verificar se a aresta entre v[i-1] e v[i] existe antes de tentar removê-la
        if G.has_edge("ve%s" % (i - 1), "ve%s" % (i)):
            G.remove_edge("ve%s" % (i - 1), "ve%s" % i)

        # Criar as arestas com o nó de bloco
        G.add_edge("ve%s" % (i - 1), "be%s" % n_branch)
        
        if(n_branch < args.n_branches):
            G.add_edge("be%s" % n_branch, "ve%s" % i)
        
        i += 1  # Incrementar o índice i após a criação do nó de bloco

    G.add_edge("be%s" %(args.n_branches), "output")
    G.add_edge("vc21", "output") #So tired, bro

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





def main(args):

    n_classes = 257

    device_str = 'cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu'

    device = torch.device(device_str)

    
    ee_model = ee_dnns.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, 
        args.dim, args.exit_type, device, args.distribution)

    ee_dnn_graph = build_edge_cloud_graph(ee_model, args.n_branches)





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

    parser.add_argument('--temp_init', type=float, default=1.0, help='Initial temperature to start the Temperature Scaling')

    parser.add_argument('--max_iter', type=float, default=10000, help='Max Interations to optimize the Temperature Scaling')

    args = parser.parse_args()

    main(args)