from dataload import *
from args import config
import warnings
from experiment import Experiment
from save import save_result
import time  
import os
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args, data):
    # fusion features
    entries = ['circRNA', 'drug'] 
    networks = dict() 
    for entrie in entries:  
        network= Experiment().MACFE(args, data, entrie)
        networks[entrie] = network

    c_d_matrix = datapro().read_csv(args.datapath + 'c_d.csv')
    c, d = c_d_matrix.shape
    original_adj = np.vstack((np.hstack((np.zeros(shape=(c, c), dtype=int), c_d_matrix)),
                              np.hstack((c_d_matrix.T, np.zeros(shape=(d, d), dtype=int)))))
    c_d_matrix = torch.FloatTensor(original_adj)
    networks['CDA'] = {'original_adj': original_adj, 'c_d_matrix': c_d_matrix}

    results, final_results = Experiment().train(args, networks)
    save_result(args, results, final_results)
    



if __name__ == "__main__":

    param = config()
    
    start_time = time.time()

    original_features = original_data(param)  ###
    main(param, original_features)

    # 结束计时
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️ 组合运行时间：{elapsed:.2f} 秒")