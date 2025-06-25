import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from wtconv import WTConv2d
from einops.layers.torch import Rearrange



class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]
    def forward(self, x, mask=False, multi_head=False):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model
        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)
        attention = self.softmax(attention / self.scaled_factor)
        if mask is True:
            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)
        return attention, embeddings


class SpatialTemporalFusion(nn.Module):
    def __init__(self, obs_len=8):
        super(SpatialTemporalFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )
        self.shortcut = nn.Sequential()
    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()


class SparseWeightedAdjacency(nn.Module):
    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, obs_len=8):
        super(SparseWeightedAdjacency, self).__init__()

        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims)

        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)

        # interaction mask
        self.spatial_wtconvolutions = WTConv2d(4, 4, kernel_size=5, wt_levels=3)
        self.temporal_wtconvolutions = WTConv2d(4, 4, kernel_size=5, wt_levels=3)

        self.spatial_output = nn.Softmax(dim=-1)
        self.temporal_output = nn.Softmax(dim=-1)

    def forward(self, graph, identity):
        assert len(graph.shape) == 3

        spatial_graph = graph[:, :, 1:]  # (T N 2)
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)

        # (T num_heads N N)   (T N d_model)
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True)
        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, multi_head=True)

        # attention fusion
        dense_spatial_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        # wavelet convolutions
        dense_spatial_interaction = self.spatial_wtconvolutions(dense_spatial_interaction)   # (T num_heads N N)
        dense_temporal_interaction = self.temporal_wtconvolutions(dense_temporal_interaction)   # (N num_heads T T)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        # self-connected
        normalized_spatial_adjacency_matrix = spatial_interaction_mask + identity[0].unsqueeze(1)
        normalized_temporal_adjacency_matrix = temporal_interaction_mask + identity[1].unsqueeze(1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix



class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()
        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()
        self.dropout = dropout
    def forward(self, graph, adjacency):
        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)
        return gcn_features  # [batch_size num_heads seq_len hidden_size]



############################################################################################
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_gap = nn.AdaptiveAvgPool2d(1)
        self.max_gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(nn.Conv2d(dim * 2, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),)
    def forward(self, x):
        x_avg_gap = self.avg_gap(x)
        x_max_gap = self.max_gap(x)
        x_gap = torch.concat([x_avg_gap, x_max_gap], dim=1)
        cattn = self.ca(x_gap)
        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
############################################################################################
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        # self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        # result = self.conv(result)
        return result
############################################################################################



class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_gcn = GraphConvolution(in_dims, embedding_dims)
        self.temporal_gcn = GraphConvolution(in_dims, embedding_dims)

        self.cga_fusion = CGAFusion(4, reduction=4)

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):
        # graph [1 seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]

        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)

        gcn_spatial_features = self.spatial_gcn(spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)
        gcn_temporal_features = self.temporal_gcn(tem_graph, normalized_temporal_adjacency_matrix)


        # gcn_spatial_features=[N, heads, T, 16]   gcn_temporal_features=[N, heads, T, 16]
        spatial_temporal_features = self.cga_fusion(gcn_spatial_features, gcn_temporal_features)

        return spatial_temporal_features


class TrajectoryModel(nn.Module):
    def __init__(self,embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency()

        # graph convolution
        self.stsgcn = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)

        self.fusion_ = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(
            nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()))

        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.output = nn.Linear(embedding_dims // num_heads, 2)
        self.multi_output = nn.Conv2d(num_heads, 20, 1, padding=0)

    def forward(self, graph, identity):
        # graph 1 obs_len N 3

        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)

        gcn_representation = self.stsgcn(graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix)

        gcn_representation = gcn_representation.permute(0, 2, 1, 3)
        # gcn_representation [Nums, 8, heads, 16]

        features = self.tcns[0](gcn_representation)
        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)
        # features=[N, Tpred, nums, 16]

        prediction = self.output(features)   # prediction=[N, Tpred, nums, 2]
        prediction = self.multi_output(prediction.permute(0, 2, 1, 3))   # prediction=[N, 20, Tpred, 2]

        return prediction.permute(1, 2, 0, 3).contiguous()   # prediction=[20, Tpred, N, 2]
