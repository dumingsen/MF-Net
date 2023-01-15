import math
import time
from functools import reduce
import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
from torch.nn import ModuleList
#from net import gtnet
visual_feature_map = False
from model.layer import *

# wa_bottom=True
# Prob = False
# mask_bottom = True   #true mask,   false 没有mask
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=self.displacement, dims=1)#沿给定维数滚动张量，移动到最后一个位置以外的元素将在第一个位置重新引入


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

def default(val, default_val):
    return val if val is not None else default_val
#dim_head = default(dim_head, dim // heads)

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# def create_mask(window_size, displacement, upper_lower, left_right):
def create_mask(window_size, displacement, MASK):
    mask = torch.zeros(window_size, window_size)
    if not MASK:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')  # 不带mask

    # 带mask
    else:
        mask[-displacement:, :-displacement] = float('-inf')
        mask[:-displacement, -displacement:] = float('-inf')
    return mask


def get_relative_distances(window_size):
    indices = torch.arange(window_size)
    distances = indices[None, :] - indices[:, None]
    return distances

activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Softmax": torch.nn.Softmax}

class Attention(nn.Module):
    def __init__(self, n_head, n_embd, win_len, scale, q_len, sub_len, sparse=None, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()

        if (sparse):
            print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        self.register_buffer('mask_tri', mask)
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if ((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while (index >= 0):
                if ((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if ((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor, activation="Softmax"):
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):

        value = self.value(x)
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0))
        query_key = self.query_key(qk_x).permute(0, 2, 1)
        query, key = query_key.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)
        attn = self.c_proj(attn)
        attn = self.resid_dropout(attn)
        return attn

class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x
# def print(*args, **kwargs):
#     flag = False
#     # flag = False
#     if flag:
#         print(*args, **kwargs)
#     else:
#         pass

class WindowAttention(nn.Module):##
    def __init__(self, t,dim, heads, head_dim, shifted, window_size, relative_pos_embedding, wa_dim, wa, prob, mask,
                 #dim=hidden_dim
                 # down_dim=length,  # length = 1536 * 2，降维维度
                 # hidden_dim=(96, 192, 62),                 # layers=(2, 2, 6, 2),                 # heads=(3, 6, 12, 24),
                 # channels=in_channel,                 # num_classes=num_class,                 # head_dim=32,
                 # window_size=args.window, 64                # downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
                 # relative_pos_embedding=True,

                 #n_head,
                 #n_embd,
                 win_len, scale1, q_len, sub_len, sparse=None, attn_pdrop=0.1, resid_pdrop=0.1,
#
        # """
        #         Args:
        #             n_time_series: Number of time series present in input
        #             n_head: Number of heads in the MultiHeadAttention mechanism
        #             seq_num: The number of targets to forecast
        #             sub_len: sub_len of the sparse attention
        #             num_layer: The number of transformer blocks in the model.
        #             n_embd: The dimention of Position embedding and time series ID embedding#n_embd + n_time_series
        #             forecast_history: The number of historical steps fed into the time series model
        #             dropout: The dropout for the embedding of the model.
        #             additional_params: Additional parameters used to initalize the attention model. Can inc
        # """

             # n_time_series: int, n_head: int, num_layer: int,
             #     n_embd: int, forecast_history: int, dropout: float, q_len: int, additional_params: Dict,
             #     activation="Softmax", forecast_length: int = None, scale_att: bool = False, seq_num1=None,
             #     sub_len=1, mu=None):
        #d = DecoderTransformer(3, 8, 4, 128, 20, 0.2, 1, {}, seq_num1=3, forecast_length=1)

         ):
        super().__init__()
        inner_dim = head_dim * heads


        self.heads = heads
        self.scale = head_dim ** -0.5

        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.prob = prob
        self.dropout = nn.Dropout(0.1)

        if self.shifted:##如果shifted
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.left_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, MASK=mask),
                                          requires_grad=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)##inner_dim = head_dim * heads

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1##相对位置索引，首先加上m-1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1))#然后行标乘以2m-1，返回一个正态分布
        else:

            #可以参与梯度，进行forward
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))#次方

        self.to_out = nn.Linear(inner_dim, dim)

        self.window_attention = nn.Linear(wa_dim, wa_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ReLU()
        self.wa = wa

##=================================================================================
        if (sparse):
            print('Activate log sparse!')
            mask = self.log_mask(win_len, sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)#
            # pytorch中tril函数主要用于返回一个矩阵主对角线以下的下三角矩阵，其它元素全部为0 00

        #register_buffer的作用是将torch.ones(2,3)这个tensor注册到模型的 buffers() 属性中，
        # 并命名为a,这代表a对应的是一个持久态，不会有梯度传播给它，但是能被模型的state_dict记录下来。可以理解为模型的常数。
        self.register_buffer('mask_tri', mask)
        n_head = heads#
        self.n_head = n_head
        n_embd=192
        self.split_size = n_embd * self.n_head#
        self.scale1 = scale1
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len)##
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.cbam=cbamblock(768,16,7)
    def forward(self, x):
        print('wondow size',self.window_size)#64
        print('windowattentin x',x.shape)#[16, 768, 192]),b n d
        if self.shifted:####
            x = self.cyclic_shift(x)
            print('shift x',x.shape)

        # b, n, t,c, h = *x.shape, self.heads
        #b, p, f, h = *x.shape, self.heads  # 32,128,96, 3     #128代表分了多少段，96是特征
#1
        b, n, d, h= *x.shape,  self.heads#batch,长度,维度，

#=====================================================================================================================
        if n <= self.window_size:
            self.window_size = n
        # b batch_size   p : patch number f: feature

        #CBAM block
        if self.wa:
            x=self.cbam(x)
            print(self.wa)
        # if self.wa:
        #     y = self.activation(self.window_attention(self.avgpool(x).reshape(b, n)))
        #     x = x * y.unsqueeze(2).repeat(1, 1, d)
        #     print(wa)

        qkv = self.to_qkv(x).chunk(3, dim=-1)#分块,tuple
        print('self.to_qkv(x)',self.to_qkv(x).shape)#[16, 768, 288]

        new_p = n // self.window_size
        # q, k1, v = map(
        #     lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
        #                         h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        ##qkv矩阵
        #map函数的原型是map(function, iterable, …)，将function应用于iterable的每一个元素，结果以列表的形式返回
        print('window',self.window_size)
        q, k1, v = map(
            lambda t: rearrange(t, 'b (new_w new_p) (head dim) -> b head new_p new_w dim',head=h, new_w=self.window_size)
            , qkv)#def rearrange(inputs, pattern, **axes_lengths) ⟶ \longrightarrow ⟶ transform_inputs
        #16, 768, 192
        #batch_size, (window_size,patch_num), (head,feature)
        # q  {batch_size,head,patch_num,window_size,feature}
        start_time = time.time()
        print('q k1 v',q.shape,k1.shape,v.shape)#[16, 3, 12, 64, 32]

        if not self.prob:###不稀疏
            print('==============================================不稀疏')
            dots = einsum('b h w i d, b h w j d -> b h w i j', q, k1) * self.scale#爱因斯坦求和,q，k
            print('dots',dots.shape)
            #([16, 3, 12, 64, 64]
            if visual_feature_map:##特征图
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(int(dots.shape[1])):
                    sns.heatmap(torch.softmax(dots[0][i][0], dim=-1).cpu().detach().numpy())
                    # plt.savefig('heatmap_{0}.pdf'.format(i), format='pdf')
                    plt.show()

            if self.relative_pos_embedding:##相对位置bias
                print('self.relative_indices[:, :]',self.relative_indices[:, :].shape)#[32, 32]
                print('self.pos_embedding[self.relative_indices[:, :].to(torch.long)]',self.pos_embedding[self.relative_indices[:, :].to(torch.long)].shape)#[32, 32])
                dots += self.pos_embedding[self.relative_indices[:, :].to(torch.long)]
                print('dots', dots.shape)
            else:
                dots += self.pos_embedding

            if self.shifted:###
                dots[:, :, -new_p:] += self.left_mask
                print('dots', dots.shape)

            attn = self.dropout(dots.softmax(dim=-1))
            print('attn', attn.shape)
            out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)#v
            print('out', out.shape)

        else:#应用稀疏
#1
            # b, n, d, h= *x.shape,  self.heads#batch,长度,维度，
            # qkv = self.to_qkv(x).chunk(3, dim=-1)#分块,tuple
            # print('self.to_qkv(x)',self.to_qkv(x).shape)#[16, 768, 288]
            #
            # new_p = n // self.window_size
            #
            # print('window',self.window_size)
            # q, k1, v = map(
            #     lambda t: rearrange(t, 'b (new_w new_p) (head dim) -> b head new_p new_w dim',head=h, new_w=self.window_size), qkv)
            # #batch_size, (window_size,patch_num), (head,feature)
            # # {batch_size,head,patch_num,window_size,feature}
            # start_time = time.time()
            # print('q k1 v',q.shape,k1.shape,v.shape)#[16, 3, 12, 64, 32]

            attn = self.attn(q, k1, v,new_p)
            #attn = self.attn(query, key, value)
            # attn = self.merge_heads(attn)
            # attn = self.c_proj(attn)
            attn = self.resid_dropout(attn)

            print('attn',attn.shape)#

#3 attention attn
            out=attn
            #out = self.resid_dropout(attn)#
        out = rearrange(out, 'b head patch window_size dim -> b (patch window_size) (head dim)',
                        head=h)
        # 16, 768, 96
        # batch_size, (window_size,patch_num), (head,feature)
        # q  {batch_size,head,patch_num,window_size,feature}
        print('out1', out.shape)  # 16, 768, 96
        out = self.to_out(out)
        print('out2', out.shape)  # 16, 768, 96 b windowsize c
        if self.shifted:##shifted
            out = self.cyclic_back_shift(out)
            print('shift cyclic_back_shift',out.shape)#[16, 768, 62]) 16, 768, 1197

        #还原原本输入
        out=ou
        return out

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view( 1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if ((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while (index >= 0):
                if ((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if ((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor, new_p,activation="Softmax"):
        activation = activation_dict[activation](dim=-1)
        # pre_att = torch.matmul(query, key)
        pre_att = einsum('b h w i d, b h w j d -> b h w i j', query, key) * self.scale
        print('pre_att', pre_att.shape)  # [16, 3, 12, 64, 64]

        #出现问题
        #=================================================================================
        if self.relative_pos_embedding:  ##相对位置bias
            print('self.relative_indices[:, :]', self.relative_indices[:, :].shape)  # 64, 64]
            print('self.pos_embedding[self.relative_indices[:, :].to(torch.long)]',self.pos_embedding[self.relative_indices[:, :].to(torch.long)].shape)  # 64, 64]
            pre_att += self.pos_embedding[self.relative_indices[:, :].to(torch.long)]
            print('relative pos embedding pre_att', pre_att.shape)#[16, 3, 12, 64, 64]

        # ===================================================================================
        #问题所在
        # if self.shifted:  ###
        #     print('-new_p',-new_p)#-12
        #     print('pre_att[:, :, -new_p:]',pre_att[:, :, -new_p:].shape)#16, 3, 12, 64, 64]
        #     print('self.left_mask',self.left_mask.shape)#[64, 64
        #     pre_att[:, :, -new_p:] += self.left_mask#
        #     print('left mask pre attn',pre_att.shape)#16, 3, 12, 64, 64]
        #===================================================================================

        if self.scale1:
            pre_att = pre_att / math.sqrt(value.size(-1))
        print('pre_att = pre_att / math.sqrt(value.size(-1))',(pre_att / math.sqrt(value.size(-1))).shape)#torch.Size([16, 3, 768, 768])#[16, 3, 12, 64, 64])
        print('pre_att.size(-2)',pre_att.size(-2))# 64

        pre_att = rearrange(pre_att, 'b head patch window_size dim -> b head (patch window_size) dim')
        print('rearrange preatt',pre_att.shape)#[16, 3, 768, 64]

        print('mask_tri',self.mask_tri.shape)# 1, 1, 64, 64])
        mask = self.mask_tri[:, :, :pre_att.size(-2), :pre_att.size(-1)]
        print('mask',mask.shape)#1, 1, 64, 64])
        pre_att = pre_att * mask + -1e9 * (1 - mask)#
        # pre_att = activation(pre_att)#([16, 768, 192]#softmax

        pre_att = rearrange(pre_att, ' b head (patch window_size) dim-> b head patch window_size dim',
                            window_size=self.window_size)
        # ===================================================================================
        # 问题所在
        if self.shifted:  ###
            print('-new_p',-new_p)#-12
            print('pre_att[:, :, -new_p:]',pre_att[:, :, -new_p:].shape)#16, 3, 12, 64, 64]
            print('self.left_mask',self.left_mask.shape)#[64, 64
            #pre_att[:, :, -new_p:] += self.left_mask#
            pre_att[:, :, -new_p:]= pre_att[:, :, -new_p:]+ self.left_mask
            print('left mask pre attn',pre_att.shape)#16, 3, 12, 64, 64]
        # ===================================================================================
        pre_att = activation(pre_att)

        pre_att = self.attn_dropout(pre_att)
        print('pre_att',pre_att.shape)#([16, 3, 12, 64, 64]
#qkv
        # pre_att = rearrange(pre_att, ' b head (patch window_size) dim-> b head patch window_size dim',window_size=self.window_size)
        attn = torch.matmul(pre_att, value)
        print('attnn', attn.shape)#[16, 3, 12, 64, 32]
        return attn

    def merge_heads(self, x):
        print('merge x',x.shape)#16, 3, 12, 64, 32
        x = x.permute(0, 2, 1, 3).contiguous()#当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)



class SwinBlock(nn.Module):
    def __init__(self, t,dim, heads, head_dim, mlp_dim, shifted,
                 window_size, relative_pos_embedding, wa_dim, wa, prob, mask,
                 ):
        super().__init__()


        ##注意力
        self.attention_block = Residual(PreNorm(dim, WindowAttention(t,
                                                                     dim=dim,#hidden
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,###区别
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
            wa_dim=wa_dim,#wa_dim=down_dim // reduce(lambda x, y: x * y,downscaling_factors[:i + 1]),#(4,2, 2,2),  # 代表多长的时间作为一个特征
                                                                     wa=wa,
                                                                     prob=prob,
                                                                     mask=mask,
                                                                     win_len=768, scale1=False, q_len=1, sub_len=6, sparse=True, attn_pdrop=0.1, resid_pdrop=0.1,
                                                                     )))
        ##
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        print('swin block x',x.shape)#[16, 768, 1197]
        x = self.attention_block(x)
        print('x',x.shape)
        x = self.mlp_block(x)#
        print('x',x.shape)
        return x

#Patch Merging层进行下采样。该模块的作用是做降采样，用于缩小分辨率，调整通道数 进而形成层次化的设计，同时也能节省一定运算量。
#patch Merging是一个类似于池化的操作，但是比Pooling操作复杂一些。池化会损失信息，patch Merging不会。
class PatchMerging(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, downscaling_factor=4):
        super(PatchMerging, self).__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor, out_channels)

    def forward(self, x):
        print('PatchMerging1 ',x.shape)#torch.Size([16, 3072, 144]) [16, 3072, 9]
        x = rearrange(x, 'b (p f) c -> b f (p c)', p=self.downscaling_factor)  # p c 是一个特征   f是代表分为了多少段 说反了
        print('PatchMerging2 ', x.shape)#([16, 768, 576]) [16, 768, 36]
        x = self.linear(x)#
        print('PatchMerging3 ', x.shape)#[16, 768, 62] [16, 768, 1197]
        return x

#一个stage便可以看做由Patch Merging和Swin Transformer Block组成
class StageModule(nn.Module):
    def __init__(self, t,in_channels, hidden_dimension, layers, downscaling_factor, num_heads,
                 head_dim, window_size, wa, prob, mask,
                 relative_pos_embedding, wa_dim=4096,
                 ):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        ##1
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)
        #2
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([

                ##swin transformer,成对使用的w-msa和sw-msa
                ##非shifted
                SwinBlock(t,dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa_dim=wa_dim, wa=wa, prob=prob, mask=mask
                          ),
                ##shifted
                SwinBlock(t,dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa_dim=wa_dim, wa=wa, prob=prob, mask=mask
            )
            ]))

    def forward(self, x):
        x = self.patch_partition(x)##分割
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)

            ##
            x = shifted_block(x)
        # return x.permute(0, 3, 1, 2)
        return x


class attentionNet(nn.Module):
    def __init__(self, hidden_dim, layers, heads, channels, wa, prob, mask,
                 t=300, down_dim=1024, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2),relative_pos_embedding=True
                 ):

        print('hidden_dim, layers, heads, channels, wa, prob, mask,'
              't, down_dim, num_classes, head_dim, window_size,'
              'downscaling_factors, relative_pos_embedding,k,dim_head, one_kv_head,share_kv',
              hidden_dim, layers, heads, channels, wa, prob, mask,
              t, down_dim, num_classes, head_dim, window_size,
              downscaling_factors)
        '''
        
        :param hidden_dim: 隐藏层个数        :param layers: 每一层的block循环次数        :param heads: head个数
        :param channels: 输入通道        :param wa: bool 是否启用window attention        :param prob: bool  是否启动prob attention
        :param mask: bool 是否启用mask        :param t: 输入的time stamp 长度        :param down_dim: 嵌入层的降维长度#
        :param num_classes: 输出类别        :param head_dim: 每个head的维度
        :param window_size: 窗口大小        :param downscaling_factors:每层下采样 倍数        :param relative_pos_embedding: 是否使用相对位置信息
        '''
        super().__init__()
        self.downsample = nn.Linear(t, down_dim)#ts长度,3072
        print('down_dim',down_dim)#3072

        self.EncoderList = ModuleList()#
        for i in range(1):#hidden_dim=(96, 192, 62),#len(hidden_dim)
            layer = StageModule(#
                t,
                in_channels=channels if i == 0 else hidden_dim[i - 1],
                hidden_dimension=t,#(96, 192, 62),#hidden_dim[i],叠加多层，最后一层必须是时间序列长度
                layers=layers[i],#(2, 2, 6, 2),
                downscaling_factor=downscaling_factors[i],
                num_heads=heads[i],##每一层的head都不同，(3, 6, 12,24),
                head_dim=head_dim,#32
                window_size=window_size,
                relative_pos_embedding=relative_pos_embedding,
                wa_dim=down_dim // reduce(lambda x, y: x * y,
                                          downscaling_factors[:i + 1]),#(4,2, 2,2),  # 代表多长的时间作为一个特征
                wa=wa,
                prob=prob,
                mask=mask

            )
            self.EncoderList.append(layer)###append


    def forward(self, ts):##ts为流入的训练数据

        print('ts',ts.shape)## torch.Size([16, 144, 62]),#16, 9, 144]
        ds = self.downsample(ts)  # B,C,L',（ts长度，3072）#
        print('ds',ds.shape)#torch.Size([16, 144, 3072])#[16, 9, 3072]
        x = ds.transpose(2, 1)
        print('transpose ds',x.shape)#torch.Size([16, 3072, 144])#[16, 3072, 9]

        for Encoder in self.EncoderList:##
            x = Encoder(x)
        encoder = x
        print('encoder',encoder.shape)#([16, 192, 62])#[16, 768, 62]


        return encoder

