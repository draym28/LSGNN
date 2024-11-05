import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from torch_scatter import scatter_mean
from torch_sparse import SparseTensor, matmul

from tqdm import tqdm



class LSGNN(nn.Module):
    """local similarity graph neural network"""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        config, 
        num_nodes, 
        ds='cora', 
        ):
        super(LSGNN, self).__init__()

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

        if num_reduce_layers == 1:
            self.reduce = [Parameter(torch.zeros([2*self.K+1, in_channels, hidden_channels]))]
        elif num_reduce_layers > 1:
            self.reduce = [Parameter(torch.zeros([2*self.K+1, in_channels, 2*hidden_channels]))]
            for _ in range(num_reduce_layers-2):
                self.reduce.append(Parameter(torch.zeros([2*self.K+1, 2*hidden_channels, 2*hidden_channels])))
            self.reduce.append(Parameter(torch.zeros([2*self.K+1, 2*hidden_channels, hidden_channels])))
        else:
            raise NotImplementedError
        self.reset_parameters()

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

    @torch.no_grad()
    def reset_parameters(self):
        for i, param in enumerate(self.reduce):
            self.register_parameter(f'reduce_{i}', param)
            nn.init.xavier_uniform_(param.data)

    def dist(self, x, edge_index:torch.Tensor):
        def _d(x, src, tgt):
            if self.method == 'cos':
                d = (x[src] * x[tgt]).sum(dim=-1)
            elif self.method == 'norm2':
                d = torch.norm(x[src] - x[tgt], p=2, dim=-1)
            return d

        split_size = 10000
        dist = []
        for ei_i in tqdm(edge_index.split(split_size, dim=-1), ncols=70):
            src_i, tgt_i = ei_i
            d = _d(x, src_i, tgt_i)
            dist.append(d)
        dist = torch.cat(dist, dim=0)

        dist = dist.view(-1, 1)
        dist = torch.cat([dist, dist.square()], dim=-1)
        return dist

    def prop(self, x, filters):
        filter_l, filter_h = filters

        # propagate first
        x = x.type(torch.float32)
        # first propagation
        x_L = x.clone()
        x_H = x.clone()
        x_L = matmul(filter_l, x_L)
        x_H = matmul(filter_h, x_H)
        out_L = [x_L]
        out_H = [x_H]
        x_L_sum = 0
        x_H_sum = 0
        # continue propagation
        for _ in range(1, self.K):
            if self.config['irdc']:
                x_L_sum = x_L_sum + out_L[-1]
                x_H_sum = x_H_sum + out_H[-1]
                x_L = matmul(filter_l, (1.-self.gamma)*x-self.gamma*x_L_sum)
                x_H = matmul(filter_h, (1.-self.gamma)*x-self.gamma*x_H_sum)
            else:
                x_L = matmul(filter_l, x_L)
                x_H = matmul(filter_h, x_H)
            out_L.append(x_L)
            out_H.append(x_H)

        x_out_L_out_H = torch.stack([x] + out_L + out_H, dim=0)
        return x_out_L_out_H

    def local_sim(self, x, edge_index, dist):
        dev = x.device
        _, tgt = edge_index
        dist = self.dist_mlp(dist).view(-1)
        return scatter_mean(
            dist.cpu(), tgt.cpu(), out=torch.zeros([x.shape[0]])).to(dev)

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
        for reduce_layer in self.reduce:
            x_out_L_out_H = torch.bmm(x_out_L_out_H, reduce_layer)
            x_out_L_out_H = F.normalize(x_out_L_out_H, p=2, dim=-1)
            x_out_L_out_H = F.relu(x_out_L_out_H)

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