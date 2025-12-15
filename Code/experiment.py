import copy
import os
from dataload import *
from model import *
from torch.utils.data import DataLoader
from model import autoEncoder, encoderLoss
from torch.optim import Adam
from dataload import netsDataset
from utils import obtain_constraints
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import numpy as np
import torch.nn as nn

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def MACFE(self, args, data, entrie):
        self.setup_seed(args.seed)
        top_rate = args.CFE_rate
        networks = data[f'{entrie}']['features']
        symbols = data[f'{entrie}']['symbols']
        if entrie == 'circRNA':
            CFE_net_nums = [args.CFE_circRNA_num for i in range(args.CFE_nets)]
            CFE_hidden_dim = args.CFE_hidden_circRNA
            CFE_batch_size = args.CFE_batch_size_circRNA
        else:
            CFE_net_nums = [args.CFE_drug_num for i in range(args.CFE_nets)]
            CFE_hidden_dim = args.CFE_hidden_drug
            CFE_batch_size = args.CFE_batch_size_drug

        constraints_ml = [np.zeros((i, i)) for i in CFE_net_nums] 
        CFE_layers = args.CFE_layers  
        models = [autoEncoder(CFE_net_nums[i], CFE_hidden_dim) for i in range(len(networks))]
        for idx_layer in range(CFE_layers): 
            emb = [np.zeros((CFE_net_nums[i], CFE_hidden_dim[idx_layer])) for i in range(args.CFE_nets)]
            reshape_xs = []
            for idx_net in range(args.CFE_nets): 
                model = models[idx_net]    
                optimizer = Adam(model.parameters(), lr=args.CFE_lr[idx_layer])
                ml = torch.from_numpy(constraints_ml[idx_net]).float().cuda() 
                criterion = encoderLoss(ml, CFE_batch_size, args.CFE_gamma) 
                if torch.cuda.is_available():
                    model = model.cuda()
                dataset = netsDataset(networks[idx_net])
                dataloader = DataLoader(dataset, batch_size=CFE_batch_size, shuffle=True)

                for epoch in range(1, args.CFE_epoch + 1):
                    model.train()
                    losses, losses_ml = [], []
                    for step, (X, y, indx) in enumerate(dataloader):
                        X, y = X.float().cuda(), y.float().cuda()
                        indx = indx
                        y_pred, _ = model(X, idx_layer)
                        loss, loss_ml = criterion(y_pred, y, indx)
                        losses.append(loss.item())
                        losses_ml.append(loss_ml.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    model.eval()
                    data = torch.from_numpy(networks[idx_net].astype(np.float32)).cuda()
                    reshape_x, features = model(data, idx_layer, flag='reshape')
                    emb[idx_net] = features.cpu().numpy()
                    reshape_xs.append(reshape_x.cpu().numpy())     

            constraints_ml = obtain_constraints(args.CFE_nets, reshape_xs, symbols, top_rate,
                                                                idx_layer)
            networks = emb
            return networks

    def loss_gcl(self, gcl_model, graph_learner, features, args, anchor_adj):
        if args.maskfeat_rate_anchor:
            mask_v1, _ = datapro().get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)
        z1, _ = gcl_model(features_v1, anchor_adj, args, branch='anchor')

        device = z1.device

        if args.maskfeat_rate_learner:
            mask, _ = datapro().get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)
        criterion = nn.CrossEntropyLoss()
        learned_adj = graph_learner(features)
        learned_adj = datapro().symmetrize(learned_adj) 
        learned_adj = datapro().normalize(learned_adj)
        z2, _ = gcl_model(features_v2, learned_adj, args, branch='learner')
     
        gcl_loss = calc_loss(z1, z2)
        loss=0
        head = get_head(256, 256).to("cuda:0")
        norm=nn.BatchNorm1d(256, affine=True).to("cuda:0")
        h1,h2=norm(z1),norm(z2)
        t1,t2=h1.argmax(dim=1),h2.argmax(dim=1)

        loss += 0.5 * (criterion(z1, t2) + criterion(z2, t1))
        return loss,learned_adj

    def evaluate_adj_by_cls(self, args,model, features, learner_adj, train_matrix):
        
       
        model.train() 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        

        if torch.cuda.is_available():
            model = model.cuda()
            features = features.cuda()

        
        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            regression_crit = Myloss()
         
            loss,drug_circ_res= self.loss_cls(model, features, learner_adj, train_matrix)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return drug_circ_res 


    def loss_cls(self, model, features, Adj, train_matrix):
        c_d_res = model(features, Adj)
        loss1 = nn.MSELoss()
        loss = loss1(c_d_res, train_matrix)
        return loss, c_d_res

    def get_metrics(self, real_score, predict_score):
        real_score = real_score.detach().numpy()
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix.dot(real_score.T)
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)

        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack(
            (recall_list, precision_list)).tolist())).T
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index]
        accuracy = accuracy_list[max_index]
        specificity = specificity_list[max_index]
        recall = recall_list[max_index]
        precision = precision_list[max_index]
        metric = [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
        metric_list = [fpr, tpr, recall_list, precision_list, auc[0, 0], aupr[0, 0]]
        return metric, metric_list
    
    def train(self, args, data):
        evaluation_results = {}
        self.setup_seed(args.seed)
        features, original_adj, c_d_matrix = datapro().get_data(data)
        num_feats = features.shape[1]
        index = datapro().datasplit(args, c_d_matrix)
        pre_matrix = np.zeros(c_d_matrix.shape)
        metric = np.zeros((1, 7))
        fpr_fold, tpr_fold, recall_fold, precision_fold, auc_fold, aupr_fold = [], [], [], [], [], []
        for fold in range(args.k_fold):
            print("------this is %d/%d cross validation ------" % (fold + 1, args.k_fold))
            train_matrix = np.matrix(c_d_matrix, copy=True)
            train_matrix[tuple(np.array(index[fold]).T)] = 0  
            # anchor view
            anchor_adj = torch.from_numpy(original_adj)
            anchor_adj = datapro().normalize(anchor_adj)
            anchor_adj = anchor_adj.float()
            out = train_matrix.shape[1]
            # graph learner——get the learner view
            graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6)

            gcl_model = MoCo_CDA(num_feats, args)
            gsl_model = GCN(num_feats, out, args)

            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_gcl = torch.optim.Adam(gcl_model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_gsl = torch.optim.Adam(gsl_model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)
            if torch.cuda.is_available():
                graph_learner = graph_learner.cuda()
                gcl_model = gcl_model.cuda()
                gsl_model = gsl_model.cuda()
                features = features.cuda()
                train_matrix = torch.FloatTensor(train_matrix).cuda()
                anchor_adj = anchor_adj.cuda()
            # start train
            for epoch in range(1, args.epochs + 1):
                graph_learner.train()  
                gcl_model.train()

                gcl_loss, Adj = self.loss_gcl(gcl_model, graph_learner, features, args, anchor_adj)

                optimizer_learner.zero_grad()
                optimizer_gcl.zero_grad()
                gcl_loss.backward()
                optimizer_learner.step()
                optimizer_gcl.step()

                learner_adj = Adj
                learner_adj = learner_adj.detach()

                if args.c:
                    anchor_adj = anchor_adj * args.tau + learner_adj * (1 - args.tau)

                if epoch == args.epochs:
                    c_d_res = self.evaluate_adj_by_cls(args,gsl_model, features, learner_adj, train_matrix)                     
                    graph_learner.eval()
                    gcl_model.eval()
                    predict_y_proba = c_d_res.reshape(args.d_num + args.c_num,
                                                      args.d_num + args.c_num).cpu().detach().numpy()
                    pre_matrix[tuple(np.array(index[fold]).T)] = predict_y_proba[tuple(np.array(index[fold]).T)]
                    real_score = c_d_matrix[tuple(np.array(index[fold]).T)]
                    pre_score = predict_y_proba[tuple(np.array(index[fold]).T)]
                    metric_tmp, metric_list_tmp = self.get_metrics(real_score, pre_score)
                    fpr_fold.append(metric_list_tmp[0])
                    tpr_fold.append(metric_list_tmp[1])
                    recall_fold.append(metric_list_tmp[2])
                    precision_fold.append(metric_list_tmp[3])
                    auc_fold.append(metric_list_tmp[4])
                    aupr_fold.append(metric_list_tmp[5])
            metric += metric_tmp
            result = {'Aupr': metric_tmp[0], 'AUC': metric_tmp[1], 'F1_Score': metric_tmp[2],
                      'ACC': metric_tmp[3], 'Recall': metric_tmp[4], 'Specificity': metric_tmp[5],
                      'Precision': metric_tmp[6]}
            evaluation_results['Fold {}'.format(fold + 1)] = result
            print('fold {}:{}'.format(fold + 1, result))
        final_result = (metric / args.k_fold)[0]
        final_results = {'Aupr': final_result[0], 'AUC': final_result[1], 'F1_Score': final_result[2],
                         'ACC': final_result[3], 'Recall': final_result[4], 'Specificity': final_result[5],
                         'Precision': final_result[6]}
        print('Average: {}'.format(final_results))



        return evaluation_results, final_results
