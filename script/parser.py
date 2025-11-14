import argparse
from dataclasses import dataclass

@dataclass
class ModelConfig:
    dim_hidden_node0: int = 40
    layer_embed_node0: int = 20
    iter_gcn: int = 5
    knode_gcn: int = 20
    kedge_gcn: int = 20
    dim_hidden_node: int = 256
    dim_hidden_edge: int = 256
    layer_embed_node: int = 2
    layer_embed_edge: int = 2
    dim_hidden_pred1: int = 128
    dim_hidden_pred2: int = 64
    layer_pred: int = 4
    fragsize: int = 9

def parse_args():
    parser = argparse.ArgumentParser(description="unZipro pre-training script")

    # --- Data group ---
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--train_list', '-t', type=str, metavar='[File]',
                            help='List of training PDBs.', required=True)
    data_group.add_argument('--valid_list', '-v', type=str, metavar='[File]',
                            help='List of validation PDBs.', required=True)
    data_group.add_argument('--pdbdir', '-pk', type=str, default='data/pdb', metavar='[DIR]',
                            help='PDB directory.')
    data_group.add_argument('--cachedir', '-cd', type=str, default='data/tmp/', metavar='[DIR]',
                            help='Cache feature directory.')

    # --- Project ---
    project_group = parser.add_argument_group('Project')
    project_group.add_argument("--project_name", default='unZipro_default', type=str,
                               help="Project name for saving model checkpoints and best model.")
    project_group.add_argument("--model", default='./Models', type=str, metavar='str',
                               help="Directory for model storage and logits.")
    project_group.add_argument("--logdir", default='./Logs', type=str, metavar='str',
                               help="Directory for logging.")
    project_group.add_argument("--config_dir", default='config', type=str, metavar='str',
                               help="Directory for model config.")
    project_group.add_argument('--logging', action='store_true',
                               help="Enable tensorboard logging during training.")
    # --- Training ---
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--batchsize', '-bs', type=int, default=10, metavar='int',
                             help='Batch size for training.')
    train_group.add_argument('--cpu', '-cpu', type=int, default=16, metavar='int',
                             help='CPU cores for data loading.')
    train_group.add_argument('--gpu', '-gpu', type=int, default=0, metavar='int',
                             help='GPU id.')
    train_group.add_argument('--learning-rate', '-lr', type=float, default=0.002, metavar='float',
                             help='Learning rate.')
    train_group.add_argument('--epochs', '-e', type=int, default=100, metavar='int',
                             help='Number of training epochs.')
    train_group.add_argument('--noise', '-ni', type=float, default=0.01, metavar='float',
                             help='Training noise.')
    train_group.add_argument('--nneighbor', '-nb', type=int, default=20, metavar='int',
                             help='Number of node neighbors.')
    
    # --- Model ---
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--dim-hidden-node0', '-dn0', type=int, default=40, metavar='int',
                         help='Hidden dimensions of the first node-embedding layers.')
    model_group.add_argument('--layer-embed-node0', '-ln0', type=int, default=20, metavar='int',
                            help='Number of the first node-embedding layers.')
    model_group.add_argument('--iter-gcn', '-ig', type=int, default=5, metavar='int',
                            help='Number of iterations of GCN layer.')
    model_group.add_argument('--knode-gcn', '-kn', type=int, default=20, metavar='int',
                            help='Additional dimension of node info for each GCN update.')
    model_group.add_argument('--kedge-gcn', '-ke', type=int, default=20, metavar='int',
                            help='Additional dimension of edge info for each GCN update.')
    model_group.add_argument('--dim-hidden-node', '-dn', type=int, default=256, metavar='int',
                            help='Hidden dimensions of the node-embedding layers.')
    model_group.add_argument('--dim-hidden-edge', '-de', type=int, default=256, metavar='int',
                            help='Hidden dimensions of the edge-embedding layers.')
    model_group.add_argument('--layer-embed-node', '-ln', type=int, default=2, metavar='int',
                            help='Number of the node-embedding layers.')
    model_group.add_argument('--layer-embed-edge', '-le', type=int, default=2, metavar='int',
                            help='Number of the edge-embedding layers.')
    model_group.add_argument('--dim-hidden-pred1', '-dp1', type=int, default=128, metavar='int',
                            help='Hidden dimensions of prediction layer 1.')
    model_group.add_argument('--dim-hidden-pred2', '-dp2', type=int, default=64, metavar='int',
                            help='Hidden dimensions of prediction layer 2.')
    model_group.add_argument('--layer-pred', '-lp', type=int, default=4, metavar='int',
                            help='Number of prediction layers.')
    model_group.add_argument('--fragsize', '-f', type=int, default=9, metavar='int',
                            help='Fragment size of prediction module.')
    return parser.parse_args()
