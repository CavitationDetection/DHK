import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='Our model options')

    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'choose device to use cpu or gpu') 

    parser.add_argument('--arch', default = 'resnet18_add_gcn', type = str, 
                        help = 'Name of model to train')
    
    parser.add_argument('--train_root', action = 'store', type = str, default = './TrainData',   
                        help = 'root path of training data')

    parser.add_argument('--train_label_path', action = 'store', type = str, default = './Label/test_label_multilabel.csv',
                        help = 'path of training data label')
    
    parser.add_argument('--test_root', action = 'store', type = str, default = './TestData',
                        help = 'root path of test data')
                        
    parser.add_argument('--test_label_path', action = 'store', type = str, default = './Label/test_label_multilabel.csv',
                        help = 'path of test data label')
    
    # params for network
    parser.add_argument('--num_classes', type=int, default = 4)

    parser.add_argument('--epoches', action = 'store', type = int, default = 10,
                        help = 'number of total epoches to run')
    
    parser.add_argument('--batch_size', action = 'store', type = int, default = 16,
                        help = 'mimi-batch size (default: 16)')
    
    parser.add_argument('--learning_rate', action = 'store', type = float, default = 0.001,
                        help = 'learning rate of the network')
    
    parser.add_argument('--n_threads', action = 'store', type = int, default = 1,
                        help = 'number of threads for multi-thread loading')
    
    parser.add_argument('--eps', action = 'store', type = float, default = 1e-8,
                        help = 'eps (default: 1e-8)')
    
    parser.add_argument('--train_batch_shuffle', action = 'store', type = bool, default = True,
                        help = 'shuffle input batch for training data')

    parser.add_argument('--test_batch_shuffle', action = 'store', type = bool, default = False,
                        help = 'shuffle input batch for training data')

    parser.add_argument('--train_drop_last', action = 'store', type = bool, default = False,
                        help = 'drop the remaining of the batch if the size does not match minimum batch size')

    parser.add_argument('--test_drop_last', action = 'store', type = bool, default = False,
                        help = 'drop the remaining of the batch if the size does not match minimum batch size')
    
    # different loss weights schemes
    parser.add_argument('--Non', action = 'store', type = str, default = 'Non',
                        help = "different weighting schemes for HTLoss or FHTLoss ('Non', 'MTH', 'CTH')")
    
    args = parser.parse_args()
    argsDict = args.__dict__
    # save pars  
    with open("pars_seeting.json",'w') as f:
        f.writelines('-----------------------start---------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('-----------------------end-----------------------' + '\n')
    return args
