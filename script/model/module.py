import torch
import torch.nn as nn
from .model_util import *

##  Graph Encoder module
class Embedding_module(nn.Module):
    """
    Graph Encoder module to embed node and edge features using iterative 1D convolutions and RGC layers.

    Args:
        nneighbor (int): Number of neighbors for central residue (default: 20).
        r_drop (float): Dropout ratio (default: 0.2).
        d_node0 (int): Initial node feature output dimension after nodefeature0 (default: 20).
        d_hidden_node0 (int): Hidden dimension for nodefeature0 (default: 40).
        nlayer_node0 (int): Number of ResBlock layers for nodefeature0 (default: 4).
        d_hidden_node (int): Hidden dimension for nodes in RGC layers (default: 256).
        d_hidden_edge (int): Hidden dimension for edges in RGC layers (default: 256).
        nlayer_node (int): Number of ResBlock layers per node in RGC (default: 2).
        nlayer_edge (int): Number of ResBlock layers per edge in RGC (default: 2).
        niter_rgc (int): Number of iterative RGC layers (default: 5).
        k_node_rgc (int): Increment for node feature dimension in RGC layers (default: 20).
        k_edge_rgc (int): Increment for edge feature dimension in RGC layers (default: 20).
        fragment_size (int): Kernel fragment size for 1D convolutions (default: 9).
    """
    def __init__(self, nneighbor, r_drop,
                 d_node0, d_hidden_node0, nlayer_node0,
                 d_hidden_node, d_hidden_edge, nlayer_node, nlayer_edge,
                 niter_rgc, k_node_rgc, k_edge_rgc, fragment_size):
        super(Embedding_module, self).__init__()
        # Fixed input dimensions
        self.d_node_in = 6    # sin(phi), cos(phi), sin(psi), cos(psi), sin(omega), cos(omega)
        self.d_edge_in = 36   # Edge features from 6x6 atom matrix
        # Kernel setup
        assert (fragment_size - 1) % 4 == 0, "(fragment_size-1) must be divisible by 4"
        kernel_size = (fragment_size - 1) // 2 + 1
        padding = (kernel_size - 1) // 2

        # If no edge layers, disable edge updates
        if nlayer_edge < 1:
            k_edge_rgc = 0

        # Node feature preprocessing: initial 1D convolution + ResBlocks
        # (num_nodes, d_node_in) -> (num_nodes, d_node0)
        self.nodefeature0 = nn.ModuleList(
            [nn.Conv1d(self.d_node_in, d_hidden_node0, kernel_size=kernel_size, stride=1, padding=padding)] +
            [ResBlock_InstanceNorm(d_hidden_node0, d_hidden_node0, dropout=r_drop) for _ in range(nlayer_node0)] +
            [ResBlock_InstanceNorm(d_hidden_node0, d_node0, dropout=r_drop)] +
            [nn.InstanceNorm1d(d_node0, affine=True)] +
            [nn.ReLU()]
        )

        # GCN Layers: iterative graph convolution
        # Each GCN layer updates nodes and edges
        # Node dims: d_node0 + i * k_node_rgc -> d_node0 + (i+1) * k_node_rgc
        # Edge dims: d_edge_in + i * k_edge_rgc -> d_edge_in + (i+1) * k_edge_rgc
        self.rgclayer = nn.ModuleList([
            RGCBlock(
                d_in=d_node0 + k_node_rgc * i,
                d_out=d_node0 + k_node_rgc * (i + 1),
                d_edge_in=self.d_edge_in + k_edge_rgc * i,
                d_edge_out=self.d_edge_in + k_edge_rgc * (i + 1),
                nneighbor=nneighbor,
                d_hidden_node=d_hidden_node,
                d_hidden_edge=d_hidden_edge,
                nlayer_node=nlayer_node,
                nlayer_edge=nlayer_edge,
                dropout=r_drop
            )
            for i in range(niter_rgc)
        ])

    def forward(self, node_in, edgemat_in, adjmat_in):
        """
        Forward pass of the embedding module.

        Args:
            node_in (torch.Tensor): Node input features, shape (num_nodes, d_node_in).
            edgemat_in (torch.Tensor): Edge input features, shape (num_nodes, num_nodes, d_edge_in).
            adjmat_in (torch.Tensor): Adjacency matrix indices for neighbors, shape (num_nodes, k_neighbors).

        Returns:
            tuple:
                node (torch.Tensor): Updated node features, shape (num_nodes, final_node_dim).
                edge (torch.Tensor): Updated edge features, shape (num_nodes, k_neighbors, final_edge_dim).
        """
        naa = node_in.size(0)
        # Extract edge features
        # Input: edgemat_in (naa, naa, 36) -> Output: edge (naa, k_neighbors, 36)
        edge = edgemat_in[adjmat_in, :].reshape(naa, -1, self.d_edge_in)
        # Node feature embedding: initial 1D conv + ResBlocks
        node = node_in.transpose(0, 1).unsqueeze(0)  # (num_nodes, d_node_in) -> (1, d_node_in, num_nodes)
        for layer in self.nodefeature0:
            node = layer(node)
        node = node.squeeze(0).transpose(0, 1)       # (1, d_node0, num_nodes) -> (num_nodes, d_node0)
        # Iterative GCN layers
        for rgc in self.rgclayer:
            node, edge = rgc(node, edge, adjmat_in)

        return node, edge

##  Graph decoder module (Iterative 1D convolution)
class Prediction_module(nn.Module):
    """
    Graph decoder module to predict node-level residue side-chain types using iterative 1D convolutions.
    
    Args:
        d_in (int): Input feature dimension per node.
        d_out (int): Output feature dimension per node (e.g., number of classes).
        d_hidden1 (int): Hidden dimension for the first convolution block.
        d_hidden2 (int): Hidden dimension for the second convolution block.
        nlayer_pred (int): Number of ResBlock layers per block.
        fragment_size (int): Size of the convolution kernel fragment; must satisfy (fragment_size-1) % 4 == 0.
        r_drop (float): Dropout rate used in ResBlock_InstanceNorm.
    """
    def __init__(self, d_in, d_out, d_hidden1, d_hidden2, nlayer_pred, fragment_size, r_drop):
        super(Prediction_module, self).__init__()
        # Check kernel fragment validity
        assert (fragment_size - 1) % 4 == 0, "fragment_size-1 must be divisible by 4"
        # Define kernel, stride, and padding for 1D convolutions
        stride = 1
        kernel_size = int((fragment_size-1)/2 + 1)
        padding = int((kernel_size-1)/2)
        # -----------------------------
        # Build the sequential module list
        # -----------------------------
        self.pred1Dconv = nn.ModuleList(
            # First conv block
            [nn.Conv1d(d_in, d_hidden1, kernel_size=kernel_size, stride=stride, padding=padding)] +
            [ResBlock_InstanceNorm(d_hidden1, d_hidden1, dropout=r_drop) for _ in range(nlayer_pred)] +
            [nn.InstanceNorm1d(d_hidden1, affine=True)] +
            [nn.ReLU()] +
            # Second conv block
            [nn.Conv1d(d_hidden1, d_hidden2, kernel_size=kernel_size, stride=stride, padding=padding)] +
            [ResBlock_InstanceNorm(d_hidden2, d_hidden2, dropout=r_drop) for _ in range(nlayer_pred)] +
            [nn.InstanceNorm1d(d_hidden2, affine=True)] +
            [nn.ReLU()] +
            # Output layer (1x1 convolution)
            [nn.Conv1d(d_hidden2, d_out, kernel_size=1, stride=1, padding=0)]
        )

    def forward(self, node_in):
        """
        Forward pass of the prediction module.

        Args:
            node_in (torch.Tensor): Input node features of shape (num_nodes, d_in).

        Returns:
            torch.Tensor: Predicted node features of shape (num_nodes, d_out).
        """
        # Transpose and add batch dimension: (num_nodes, d_in) -> (1, d_in, num_nodes)
        node_out = node_in.transpose(0, 1).unsqueeze(0)

        # Iteratively apply conv + resblocks + norm + activation
        for layer in self.pred1Dconv:
            node_out = layer(node_out)

        # Remove batch dimension and transpose back: (1, d_out, num_nodes) -> (num_nodes, d_out)
        node_out = node_out.squeeze(0).transpose(0, 1)

        return node_out