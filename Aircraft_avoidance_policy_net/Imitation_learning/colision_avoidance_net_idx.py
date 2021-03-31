import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from custom_train_tools import CustomDataset, train_model, test_model, system_log

# parsing user input option
parser = argparse.ArgumentParser(description='Train Implementation')
parser.add_argument('--num_layers', nargs='+', type=int,
                    default=[2, 2, 2], help='num layers')
parser.add_argument('--num_nodes', nargs='+', type=int,
                    default=[40, 40, 40], help='num nodes')
parser.add_argument('--index', type=int, default=0, help='index(gpu number)')
args = parser.parse_args()

# designate GPU device number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.index)

# main loop
if __name__ == "__main__":
    # setting batch size and learning rate
    batch_size = 300
    lr = 0.001

    # setting number of layers and nodes by user input
    num_layers = args.num_layers
    _nodes = args.num_nodes
    idx = args.index

    # setting saving name of logfile, image and model weight
    model_char = "{}_{}_{}_{}_{}_{}_{}".format(
        _nodes[0], _nodes[1], _nodes[2], num_layers[0], num_layers[1], num_layers[2], idx)

    # setting log file path and system logger
    log_file = './res_log/'+model_char+'.txt'
    system_logger = system_log(log_file)

    # load train, validation dataset
    train_dataset = CustomDataset('norm_data_train_uniform_ext.csv')
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=60, drop_last=True)
    val_dataset = CustomDataset('norm_data_test_uniform_ext.csv')
    val_loader = DataLoader(dataset=val_dataset, pin_memory=True,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=60, drop_last=True)
    print("Data load complete")

    # train model
    model = train_model(num_layers, _nodes, lr, batch_size, train_loader, val_loader, model_char, system_logger)

    # test model
    for choose_ht, cmd_char_i in enumerate(["Down", "Up", "Stay"]):
        test_model(cmd_char_i, choose_ht, model, model_char, system_logger)
