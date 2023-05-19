import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor, matmul
from tqdm import tqdm

import utils as u


class BRBlock(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super(BRBlock, self).__init__()

        self.w = Parameter(torch.zeros([batch_size, in_dim, out_dim]))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.w.data)

    def forward(self, x):
        # x: (bs, N, in_dim)
        x = torch.bmm(x, self.w)
        x = F.normalize(x, p=2, dim=-1)
        x = F.relu(x)
        return x


class BatchReduce(nn.Module):
    """reduce dimension"""
    def __init__(
        self, batch_size, in_channels, hidden_channels, 
        num_layers=1,  # 1 or 2
    ):
        super(BatchReduce, self).__init__()

        self.L = num_layers

        if num_layers == 1:
            self.reduce = BRBlock(batch_size, in_channels, hidden_channels)
        elif num_layers >= 2:
            reduce = [BRBlock(batch_size, in_channels, 2*hidden_channels)]
            for _ in range(1, num_layers-1):
                reduce.append(BRBlock(batch_size, 2*hidden_channels, 2*hidden_channels))
            reduce.append(BRBlock(batch_size, 2*hidden_channels, hidden_channels))
            self.reduce = nn.Sequential(*reduce)
    
    def forward(self, x):
        return self.reduce(x)


class LSGNN(MessagePassing):
    """local similarity graph neural network"""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        config, 
        num_nodes, 
        ds='cora', 
        ):
        super(LSGNN, self).__init__(aggr='add')

        hidden_channels = config['hidden_channels']
        num_reduce_layers = config['num_reduce_layers']
        self.ds = ds
        self.K = config['K']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.method = config['method']
        self.dp = config['dropout']
        self.config = config

        self.dist_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels), 
            nn.SiLU(), 
            nn.Linear(hidden_channels, 1)
        )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels), 
            nn.SiLU(), 
            nn.Linear(hidden_channels, 3*self.K)
        )

        self.reduce = BatchReduce(2*self.K+1, in_channels, hidden_channels, num_reduce_layers)

        if config['A_embed']:
            self.A_mlp = nn.Sequential(
                nn.Linear(num_nodes, hidden_channels), 
                nn.BatchNorm1d(hidden_channels), 
                nn.ReLU(), 
            )
            final_nz = self.K + 2
        else:
            self.A_mlp = None
            final_nz = self.K + 1

        if config['out_mlp']:
            self.out_linear = nn.Sequential(
                nn.Linear(final_nz*hidden_channels, 2*hidden_channels), 
                nn.BatchNorm1d(2*hidden_channels), 
                nn.ReLU(), 
                nn.Linear(2*hidden_channels, out_channels)
            )
        else:
            self.out_linear = nn.Linear(final_nz*hidden_channels, out_channels)

        self.cache = None

    def dist(self, x, edge_index:torch.Tensor):

        split_size = 10000
        dist = []
        for ei_i in tqdm(edge_index.split(split_size, dim=-1), ncols=70):
            src_i, tgt_i = ei_i

            if self.method == 'cos':                    
                dist.append((x[src_i] * x[tgt_i]).sum(dim=-1))
            elif self.method == 'norm2':
                dist.append(torch.norm(x[src_i] - x[tgt_i], p=2, dim=-1))

        dist = torch.cat(dist, dim=0)

        dist = dist.view(-1, 1)
        dist = torch.cat([dist, dist.square()], dim=-1)
        return dist

    def local_sim(self, x, edge_index, dist=None):
        if dist is None:
            dist = self.dist(x, edge_index)

        _, tgt = edge_index
        dist = self.dist_mlp(dist).view(-1)
        return scatter_mean(dist, tgt, out=torch.zeros([x.shape[0]], device=x.device))

    def prop(self, x, edge_index, DAD=None):
        N, _ = x.shape
        dev = x.device

        # cal filters
        if DAD == None:
            DAD = u.cal_filter(dge_index=edge_index, num_nodes=N).to(dev)

        # beta
        I_indices = torch.LongTensor([[i,i] for i in range(N)]).t()
        I_values = torch.tensor([1. for _ in range(N)])
        I_sparse = torch.sparse.FloatTensor(indices=I_indices, values=I_values, size=torch.Size([N,N])).to(dev)
        filter_l = SparseTensor.from_torch_sparse_coo_tensor(self.beta * I_sparse + DAD)
        filter_h = SparseTensor.from_torch_sparse_coo_tensor((1. - self.beta) * I_sparse - DAD)

        # propagate first
        x = x.type(torch.float32)
        # first propagation
        x_L = x.clone()
        x_H = x.clone()
        x_L = self.propagate(edge_index=filter_l, x=x_L)
        x_H = self.propagate(edge_index=filter_h, x=x_H)
        out_L = [x_L]
        out_H = [x_H]
        x_L_sum = 0
        x_H_sum = 0
        # continue propagation
        for _ in range(1, self.K):
            x_L_sum = x_L_sum + out_L[-1]
            x_H_sum = x_H_sum + out_H[-1]
            x_L = self.propagate(
                edge_index=filter_l, x=(1-self.gamma)*x-self.gamma*x_L_sum)
            x_H = self.propagate(
                edge_index=filter_h, x=(1-self.gamma)*x-self.gamma*x_H_sum)

            out_L.append(x_L)
            out_H.append(x_H)

        x_out_L_out_H = torch.stack([x] + out_L + out_H, dim=0)
        return x_out_L_out_H

    def forward(self, x, edge_index, dist, x_out_L_out_H):
        N, _ = x.shape
        dev = x.device

        # cal node sim
        local_sim = self.local_sim(x, edge_index, dist)
        ls_ls2 = torch.cat([local_sim.view(-1, 1), local_sim.view(-1, 1).square()], dim=-1)

        # cal alpha
        alpha = self.alpha_mlp(ls_ls2)  # (N, 3K)
        stack_alpha = alpha.reshape([N, self.K, 3])
        alpha_I = stack_alpha[:,:,0].t().unsqueeze(-1)
        alpha_L = stack_alpha[:,:,1].t().unsqueeze(-1)
        alpha_H = stack_alpha[:,:,2].t().unsqueeze(-1)

        # reduce dimensional
        x_out_L_out_H = self.reduce(x_out_L_out_H)

        x = x_out_L_out_H[0,:,:]               # (N, hdim)
        out_I = x.expand(self.K, -1, -1)       # (K, N, hdim)
        out_L = x_out_L_out_H[1:self.K+1,:,:]  # (K, N, hdim)
        out_H = x_out_L_out_H[self.K+1:,:,:]   # (K, N, hdim)

        # fusion: (K, N, hdim)
        out = alpha_I * out_I + alpha_L * out_L + alpha_H * out_H

        # embedding A and concat representations
        if self.A_mlp is not None:
            A = SparseTensor(
                row=edge_index[0], col=edge_index[1], 
                value=torch.ones([edge_index.shape[1]]).to(dev), 
                sparse_sizes=[N,N]).to_torch_sparse_coo_tensor()
            A = self.A_mlp(A)
            out = torch.cat([x.unsqueeze(0), out, A.unsqueeze(0)], dim=0)
        else:
            out = torch.cat([x.unsqueeze(0), out], dim=0)

        # norm (K+1, N, hdim)
        if self.config['out_norm']:
            out = F.normalize(out, p=2, dim=-1)
        out = F.dropout(out, self.dp, self.training)
        out = out.permute(1, 0, 2).reshape(N, -1)

        # prediction
        out = self.out_linear(out)
        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)