import torch
import torch.nn as nn
from .module import Embedding_module, Prediction_module

##  Main module
class unZipro(nn.Module):
    def __init__(self, model_config):
        super(unZipro, self).__init__()
        self.model_config = model_config
        ##  embedding module  ##
        self.embedding = Embedding_module(nneighbor=model_config.nneighbor,
                                          r_drop=model_config.r_drop,
                                          d_node0=model_config.d_embed_node0,
                                          fragment_size=model_config.fragment_size0,
                                          d_hidden_node0=model_config.d_embed_h_node0,
                                          nlayer_node0=model_config.nlayer_embed_node0,
                                          d_hidden_node=model_config.d_embed_h_node,
                                          d_hidden_edge=model_config.d_embed_h_edge,
                                          nlayer_node=model_config.nlayer_embed_node,
                                          nlayer_edge=model_config.nlayer_embed_edge,
                                          k_node_rgc=model_config.k_node_rgc,
                                          k_edge_rgc=model_config.k_edge_rgc,
                                          niter_rgc=model_config.niter_embed_rgc)
        ##  prediction module  ##
        self.prediction = Prediction_module(d_in=model_config.d_embed_node0 + model_config.k_node_rgc * model_config.niter_embed_rgc,
                                            d_out=model_config.d_pred_out,
                                            r_drop=model_config.r_drop,
                                            d_hidden1=model_config.d_pred_h1,
                                            d_hidden2=model_config.d_pred_h2,
                                            nlayer_pred=model_config.nlayer_pred,
                                            fragment_size=model_config.fragment_size)
    def size(self):
        params = 0
        for p in self.parameters():
            if p.requires_grad:
                params += p.numel()
        return params
        
    def forward(self, node_in, edgemat_in, adjmat_in):
        # embedding
        latent, _ = self.embedding(node_in, edgemat_in, adjmat_in)
        # prediction
        out = self.prediction(latent)
        # output
        return out

    def get_embedding(self, node_in, edgemat_in, adjmat_in):
        return self.embedding(node_in, edgemat_in, adjmat_in)