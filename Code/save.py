import os


def save_result(args, results, final_results):
    # create directory
    result_savepath = args.savepath
    if not os.path.exists(result_savepath):
        os.makedirs(result_savepath)
    file = result_savepath + '/results.txt'
    with open(file, 'a') as f:
        f.write("dropedge_rate:{}, m: {}, maskfeat_rate_anchor:{},maskfeat_rate_learner:{},epochs：{}，epochs_cls：{}，CFE_epoch:{},CFE_layers：{}\n".
                format(args.dropedge_rate,args.m,args.maskfeat_rate_anchor,args.maskfeat_rate_learner, args.epochs,args.epochs_cls,args.CFE_epoch,args.CFE_layers))
        for fold, evaluation in results.items():
            f.write(
                '{}:\tAupr:{:.5f}\tAUC:{:.5f}\tF1_Score:{:.5f}\tACC:{:.5f}\tRecall:{:.5f}\t'
                'Specificity:{:.5f}\tPrecision:{:.5f}\n'.format(
                    fold, evaluation["Aupr"], evaluation["AUC"], evaluation["F1_Score"], evaluation["ACC"],
                    evaluation["Recall"], evaluation["Specificity"], evaluation["Precision"]))

        f.write('Final result:\tAupr:{:.5f}\tAUC:{:.5f}\tF1_Score:{:.5f}\tACC:{:.5f}\tRecall:{:.5f}\t'
                'Specificity:{:.5f}\tPrecision:{:.5f}\n'.format(final_results["Aupr"], final_results["AUC"],
                                                                final_results["F1_Score"], final_results["ACC"],
                                                                final_results["Recall"],
                                                                final_results["Specificity"],
                                                                final_results["Precision"]))
