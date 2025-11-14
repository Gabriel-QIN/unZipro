import torch
import torch.nn as nn

##  Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('InstanceNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        m.bias.data.fill_(0)

## Graph Convolution Block (DenseNet-like update)
class RGCBlock(nn.Module):
    def __init__(self, d_in, d_out, d_edge_in, d_edge_out, nneighbor,
                 d_hidden_node, d_hidden_edge, nlayer_node, nlayer_edge, dropout):
        super(RGCBlock, self).__init__()
        # node 20 40; edge 36 56; neighbor 20; hidden node 256; hidden edge 256; layer_node 2; layer_edge 2; dropout 0.2
        self.nlayer_edge = nlayer_edge
        self.d_in = d_in
        self.d_out = d_out
        self.k_node = d_out - d_in
        self.k_edge = d_edge_out - d_edge_in
        self.nneighbor = nneighbor
        self.d_hidden_att = 2
        #  edge update layer
        # Conv1: 36+20+20 --->>> hidden; ResBlock_bn * 2: in=hidden, out=hidden; ResBlock: hidden -> k_edge(20); bn, relu. Totaly 6 layers.
        if(nlayer_edge > 0):
            self.edgeupdate = nn.ModuleList(
                [nn.Conv1d(d_edge_in+d_in+d_in, d_hidden_edge, kernel_size=1, stride=1, padding=0)] +
                [ResBlock_BatchNorm(d_hidden_edge, d_hidden_edge, dropout=dropout) for _ in range(nlayer_edge)] +
                [ResBlock_BatchNorm(d_hidden_edge, self.k_edge, dropout=dropout)] +
                [nn.BatchNorm1d(self.k_edge, affine=True)] +
                [nn.ReLU()]
            )
        #  graph convolution layer
        #  Conv1: in = 56 + 40; out hidden; ResBlock_bn * 2: in = 40, out = 40; bn; relu. Totally 5 layers.
        self.encoding = nn.ModuleList(
            [nn.Conv1d(d_edge_out+2*d_in, d_hidden_node, kernel_size=1, stride=1, padding=0)] +
            [ResBlock_BatchNorm(d_hidden_node, d_hidden_node, dropout=dropout) for _ in range(nlayer_node)] +
            [nn.BatchNorm1d(d_hidden_node, affine=True)] +
            [nn.ReLU()]
        )
        #  residual update layer
        #  transform feature dimension `hidden_node` to  `k_node`
        self.residual = nn.ModuleList(
            [ResBlock_BatchNorm(d_hidden_node, d_hidden_node, dropout=dropout) for _ in range(nlayer_node)] +
            [ResBlock_BatchNorm(d_hidden_node, self.k_node, dropout=dropout)] +
            [nn.BatchNorm1d(self.k_node, affine=True)] +
            [nn.ReLU()]
        )

    def forward(self, x, edgevec, adjmat):
        naa = adjmat.size()[0]
        # node-vec
        node_expand = x.unsqueeze(0).expand(naa, naa, self.d_in)
        nodetrg = node_expand[adjmat, :].reshape(naa, -1, self.d_in)
        nodesrc = x.unsqueeze(1).expand(naa, self.nneighbor, self.d_in)
        ## edge update ##
        # concat node-vec & edge-vec
        if(self.nlayer_edge > 0):
            selfnode = x.unsqueeze(1).expand(naa, self.nneighbor, self.d_in)
            nen = torch.cat((selfnode, edgevec, nodetrg), 2).transpose(1, 2)
            for f in self.edgeupdate:
                nen = f(nen)
            edgevec_new = nen.transpose(1, 2)
            edgevec = torch.cat((edgevec, edgevec_new), 2)
        ## node update ##
        nodeedge = torch.cat((nodesrc, edgevec, nodetrg), 2).transpose(1, 2)
        # encoding layer
        encoded = nodeedge
        for f in self.encoding:
            encoded = f(encoded)
        aggregated = encoded.sum(2)
        residual = aggregated.unsqueeze(2)
        for f in self.residual:
            residual = f(residual)
        residual = residual.squeeze(2)
        # add dense connection
        out = torch.cat((x, residual), 1)
        # return
        return out, edgevec
        
##  ResBlock with InstanceNormalization
class ResBlock_InstanceNorm(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super(ResBlock_InstanceNorm, self).__init__()
        #  layer1
        self.bn1 = nn.InstanceNorm1d(d_in, affine=True)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0)
        #  layer2
        self.bn2 = nn.InstanceNorm1d(d_out, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_out, d_out, kernel_size=1, stride=1, padding=0)
        #  shortcut
        self.shortcut = nn.Sequential()
        if d_in != d_out:
            self.shortcut.add_module('bn', nn.InstanceNorm1d(d_in, affine=True))
            self.shortcut.add_module('conv', nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout2(self.relu2(self.bn2(out))))
        out += self.shortcut(x)
        return out


##  ResBlock with BatchNormalization
class ResBlock_BatchNorm(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super(ResBlock_BatchNorm, self).__init__()
        #  layer1
        self.bn1 = nn.BatchNorm1d(d_in, affine=True)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0)
        #  layer2
        self.bn2 = nn.BatchNorm1d(d_out, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_out, d_out, kernel_size=1, stride=1, padding=0)
        #  shortcut
        self.shortcut = nn.Sequential()
        if d_in != d_out:
            self.shortcut.add_module('bn', nn.BatchNorm1d(d_in, affine=True))
            self.shortcut.add_module('conv', nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout2(self.relu2(self.bn2(out))))
        out += self.shortcut(x)
        return out

