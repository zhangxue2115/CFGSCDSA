import csv
import random

import torch

EOS = 1e-10

import pandas as pd
from torch.utils.data import Dataset
import numpy as np

def original_data(args):
    original_features = dict()
    data_path = args.datapath
    circ_networks, circ_symbols = [], []
    circ_paths = args.circRNA
    for path in circ_paths:
        sim = pd.read_csv(data_path + f"{path}.csv", header=None).values
        circ_networks.append(sim)
        pf = pd.read_csv(data_path + f"circRNA_name.csv")
        sy = pf.values[:, 0]
        circ_symbols.append(sy)
    circRNA = dict()
    circRNA['features'] = circ_networks
    circRNA['symbols'] = circ_symbols
    dis_networks, dis_symbols = [], []
    dis_paths = args.drug
    for path in dis_paths:
        sim = pd.read_csv(data_path + f"{path}.csv", header=None).values
        dis_networks.append(sim)
        pf = pd.read_csv(data_path + f"drug_name.csv")
        sy = pf.values[:, 0]
        dis_symbols.append(sy)
    drug = dict()
    drug['features'] = dis_networks
    drug['symbols'] = dis_symbols
    original_features['circRNA'] = circRNA
    original_features['drug'] = drug
    return original_features



def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader] 
        return torch.Tensor(cd_data)

class netsDataset(Dataset):
    def __init__(self, net):
        super(netsDataset, self).__init__()
        self.net = net

    def __len__(self):
        return len(self.net)

    def __getitem__(self, item):
        x = self.net[item]
        y = self.net[item]
        return x, y, item


class datapro:
    def __init__(self):
        super(datapro, self).__init__()

    def read_csv(self, filename):
        with open(filename, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            cd_data = []
            cd_data += [[float(i) for i in row] for row in reader]
            return torch.Tensor(cd_data)

    def get_edge_index(self, matrix):
        edge_index = [[], []]
        for i in range(matrix.size(0)):
            for j in range(matrix.size(1)):
                if matrix[i][j] != 0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        return torch.LongTensor(edge_index)

    def get_data(self, data):  
        circRNA = data['circRNA']
        drug = data['drug']
        attributes_list = []

        for i in range(len(circRNA)):
            c_row, c_colum = circRNA[i].shape   
            d_row, d_colum = drug[i].shape
            attributes_list.append(np.vstack((np.hstack((circRNA[i], np.zeros(shape=(c_row, d_colum), dtype=int))),
                                              np.hstack((np.zeros(shape=(d_row, c_colum), dtype=int), drug[i])))))
        features = np.hstack(attributes_list)   
        features = features.astype(float)
        features = torch.FloatTensor(features)
    
        original_adj = data['CDA']['original_adj']
        c_d_matrix = data['CDA']['c_d_matrix']
        return features, original_adj, c_d_matrix

    def load_data(self, args, data):
        features = data['features']['features']
        original_adj = data['CDA']['original_adj']
        c_d_matrix = data['CDA']['c_d_matrix']
        return features, original_adj, c_d_matrix

    def random_index(self, index_matrix, args, sample_size=None):
        association_num = index_matrix.shape[1]
        random_index = index_matrix.T.tolist()
        random.shuffle(random_index)
        if sample_size is not None:
            random_index = random.sample(random_index, sample_size)  
            association_num = sample_size 
            
        k_folds = args.k_fold
        CV_size = int(association_num / k_folds)
        temp = np.array(random_index[:association_num - association_num % k_folds]).reshape(k_folds, CV_size,
                                                                                            -1).tolist()
        temp[k_folds - 1] = temp[k_folds - 1] + random_index[
                                                association_num - association_num % k_folds:]
        return temp

    
    def datasplit(self, args, c_d_matrix):
        pos_index_matrix = np.mat(np.where(c_d_matrix == 1))  
        neg_index_matrix = np.mat(np.where(c_d_matrix == 0))

        pos_index = self.random_index(pos_index_matrix, args)
        neg_index = self.random_index(neg_index_matrix, args, sample_size=pos_index_matrix.shape[1])

        index = [pos_index[i] + neg_index[i] for i in range(args.k_fold)]
        return index

    def normalize(self, adj):
        inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]



    def get_feat_mask(self, features, mask_rate):
        feat_node = features.shape[1]
        mask = torch.zeros(features.shape)
        samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
        mask[:, samples] = 1
        return mask.cuda(), samples

    def symmetrize(self, adj):  
        return (adj + adj.T) / 2
