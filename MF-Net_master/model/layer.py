from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


# def print(*args, **kwargs):
#     flag = False
#     # flag = False
#     if flag:
#         print(*args, **kwargs)
#     else:
#         pass


def printt(*args, **kwargs):
    flag = True
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias).to(torch.device("cuda"))

    def forward(self, x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


##图卷积
class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        print('mixprop')
        adj = adj + torch.eye(adj.size(0)).to(x.device)#.to(torch.device("cpu"))
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):#2
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class DiffPool(nn.Module):

    def __init__(self, conv_channels, residual_channels, gcn_depth, dropout, propalpha, device="cuda:0",
                 final_layer=False):
        super(DiffPool, self).__init__()
        self.device = device

        # self.embed = GraphSAGE(self.feature_size, self.feature_size, device=self.device)
        self.embed1 = mixprop(conv_channels, conv_channels, gcn_depth, dropout, propalpha)#
        # self.pool = GraphSAGE(self.feature_size, self.output_dim, device=self.device)
        self.pool1 = mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)#
        self.final_layer = final_layer

    def forward(self, x, a):
        embed = self.embed1(x, a) + self.embed1(x, a.transpose(1, 0))
        print('diff pool embed',embed.shape)#[16, 32, 6, 3072])
        pool= self.pool1(x, a) + self.pool1(x, a.transpose(1, 0))
        print('diff pool pool', pool.shape)#[16, 32, 6, 3072])
        z = embed#embed(x, a)
        if self.final_layer:
            s = torch.ones(x.size(0), self.residual_channels, device=self.device)
            print('s1',s.shape)
        else:
            s = F.softmax(pool, dim=2)#[16, 32, 6, 3072])
            print('s',s.shape)

        # # update pooled features and adj matrix
        # x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)  # 更新x
        #
        # adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor  # 更新adj

        #x_new = s @ z#.t()
        x_new = torch.matmul(s.transpose(-1, -2), z) # .t()
        print('diff pool x_new', x_new.shape)
        a_new = s@ a @ s#.t()
        print('diff pool a_new', a_new.shape)
        return x_new, a_new


class dy_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        # adj = adj + torch.eye(adj.size(0)).to(x.device)
        # d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):  ##空洞卷积
    def __init__(self, cin, cout, dilation_factor=2):  ##32，32，1
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
        # filters: 整数，输出空间的维度 （即卷积中滤波器的数量）。
        # kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
        # strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。
        # padding: "valid" 或 "same" (大小写敏感)。   valid padding就是不padding，而same padding就是指padding完尺寸与原来相同
        # 图像识别一般来说都要padding，尤其是在图片边缘的特征重要的情况下。padding多少取决于我们需要的输出是多少
        # dilation_rate: 一个整数或 2 个整数的元组或列表，dilation_rate: 一个整数或 2 个整数的元组或列表

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)

        else:  # 空
            self.emb1 = nn.Embedding(nnodes, dim)
            ##torch.nn.Embedding： 随机初始化词向量，词向量值在正态分布N(0,1)中随机取值。
            # num_embeddings, – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
            # embedding_dim,– 嵌入向量的维度，即用多少维来表示一个符号。
            # padding_idx=None,– 填充id，比如，输入长度为100，但是每次的句子长度并不一样，
            # 后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）

            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))  # 全连接
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))  # 相乘，转置
        adj = F.relu(torch.tanh(self.alpha * a))  # 邻接矩阵
        # F.relu()是函数调用，一般使用在forward函数里。而nn.ReLU()是模块调用，一般在定义网络层的时候使用。
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        # tensor和numpy都是矩阵，前者能在GPU上运行，后者只能在CPU运行，所以要注意数据类型的转换
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k,
                                                          1)  # 按行,出错，RuntimeError: selected index k out of range
        # 返回与 input 大小相同的张量，该张量由区间上均匀分布的随机数填充[0, 1) 。 torch.rand_like(input) 等效于
        # torch.rand(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)
        # 沿给定dim维度返回输入张量input中 k 个最大值。
        # 如果不指定dim，则默认为input的最后一维。
        # 如果为largest为 False ，则返回最小的 k 个值。
        mask.scatter_(1, t1, s1.fill_(1))
        # 不同之处在于 scatter() 不会直接修改原来的 Tensor，而 scatter_() 会在原来的基础上对Tensor进行修改。
        # scatter(dim, index, src)将src中数据根据index中的索引按照dim的方向进行填充。
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


##
class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        # (residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline)

        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):  # isinstance() 函数来判断一个对象是否是一个已知的类型,比如：
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:  #
            # BN（BatchNorm）和LN（LayerNorm）是两种最常用的Normalization的方法，它们都是将输入特征转换为均值为1，方差为0的数据，它们的形式是
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


# SE通道注意力机制,二维改一维
class SELayer(nn.Module):
    def __init__(self, channel, reduction=32):  # 传入输入通道数，缩放比例
        super(SELayer, self).__init__()
        # 对输入信号，提供1维的自适应平均池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。
        # 最后一维度
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 平均池化高宽为1
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b,c,h.w
        b, c, _ = x.size()  # batch \channel\ high\ weight
        # b,c,1,1----> b,c
        y = self.avg_pool(x).view(b, c)  # 调整维度、去掉最后两个维度
        print('y', y.shape)
        # b,c- ----> b,c/16 ---- >b,c ----> b,c,1,1
        y1 = self.fc1(y).view(b, c, 1)  # 添加上h,w维度
        print('y1', y1.shape)

        # # b,c,1,1----> b,c
        # z = self.avg_pool(x)  # 平均欧化
        # print('z',z.shape)
        # # b,c- ----> b,c/16 ---- >b,c
        # y2 = self.fc2(z)  # 降维、升维
        # print('y2',y2.shape)

        return x * y1.expand_as(x)  # 来扩展张量中某维数据的尺寸，将输入tensor的维度扩展为与指定tensor相同的size


# CBMA  通道注意力机制和空间注意力机制的结合
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 平均池化高宽为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 最大池化高宽为1

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # 加和操作
        return self.sigmoid(out)  # sigmoid激活操作


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w,返回索引
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        return self.sigmoid(x)  # sigmoid激活操作


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
        return x


# selfattention,无改进版本
class Attention_Layer(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs, lens):

        size = inputs.size()
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
        V = self.V_linear(inputs)

        # 还要计算生成mask矩阵
        max_len = max(lens)  # 最大的句子长度，生成mask矩阵
        sentence_lengths = torch.Tensor(lens)  # 代表每个句子的长度
        mask = torch.arange(sentence_lengths.max().item())[None, :] < sentence_lengths[:, None]
        mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        mask = mask.expand(size[0], max_len, max_len)  # [batch_size, max_len, max_len]

        # print('\nmask is :', mask.size())

        # 下面生成用来填充的矩阵
        padding_num = torch.ones_like(mask)
        padding_num = -2 ** 31 * padding_num.float()

        # print('\npadding num is :', padding_num.size())

        # 下面开始计算啦
        alpha = torch.matmul(Q, K)

        # 下面开始mask
        alpha = torch.where(mask, alpha, padding_num)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)
        # print('\nalpha is :', alpha)

        out = torch.matmul(alpha, V)

        return out


if __name__ == '__main__':
    out = torch.rand(3, 10, 128)  # 这里假设是RNN的输出，维度分别是[batch_size, max_len, hidden_size * 2]
    att_L = Attention_Layer(64, True)  # 参数分别是 hidden_size, 双向RNN：True
    lens = [7, 10, 4]  # 一个batch文本的真实长度

    att_out = att_L(out, lens)  # 开始计算

# multihead attention


# swin transformer
