from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_embed_node0: int = 20
    d_embed_h_node0: int = 40
    nlayer_embed_node0: int = 4
    niter_embed_rgc: int = 5
    k_node_rgc: int = 20
    k_edge_rgc: int = 20
    d_embed_h_node: int = 256
    d_embed_h_edge: int = 256
    nlayer_embed_node: int = 2
    nlayer_embed_edge: int = 2
    fragment_size: int = 9
    d_pred_h1: int = 128
    d_pred_h2: int = 64
    nlayer_pred: int = 4
    d_pred_out: int = 20
    dist_chbreak: float = 2.0
    dist_mean: float = 6.4
    dist_var: float = 2.4
    fragment_size0: int = 9
    nneighbor: int = 20
    r_drop: float = 0.2

@dataclass
class Config:
    nepoch: int = 100
    learning_rate: float = 0.002
    dist_chbreak: float = 2.0
    dist_mean: float = 6.4
    dist_var: float = 2.4
    nneighbor: int = 20
    d_pred_out: int = 20
    r_drop: float = 0.2
    fragment_size0: int = 9
    d_embed_node0: int = 20
    d_embed_h_node0: int = 40
    nlayer_embed_node0: int = 4
    niter_embed_rgc: int = 5
    k_node_rgc: int = 20
    k_edge_rgc: int = 20
    d_embed_h_node: int = 256
    d_embed_h_edge: int = 256
    nlayer_embed_node: int = 2
    nlayer_embed_edge: int = 2
    fragment_size: int = 9
    d_pred_h1: int = 128
    d_pred_h2: int = 64
    nlayer_pred: int = 4
    train_list: str = ""
    valid_list: str = ""
    project_name: str = "unZipro_default"
    model_dir: str = "Models"
    logdir: str = "Logs"
    pkldir: str = "tmp/pkl"

def update_config_from_args(config: Config, args):
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config