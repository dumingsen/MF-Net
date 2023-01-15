import time
from functools import reduce
import numpy as np
import torch
from einops import rearrange
from torch import nn, einsum
from torch.nn import ModuleList
#from net import gtnet
visual_feature_map = False


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
    print('indices',indices.shape)#torch.Size([64])
    distances = indices[None, :] - indices[:, None]
    print('distances',distances.shape)#e([64, 64])
    return distances


class WindowAttention(nn.Module):##
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, wa_dim, wa, prob, mask):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.prob = prob
        self.dropout = nn.Dropout(0.1)

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.left_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, MASK=mask),
                                          requires_grad=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1))##
        else:

            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

        self.window_attention = nn.Linear(wa_dim, wa_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ReLU()
        self.wa = wa

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        # b, n, t,c, h = *x.shape, self.heads
        b, p, f, h = *x.shape, self.heads  # 32,128,96, 3     #128代表分了多少段，96是特征
        if p <= self.window_size:
            self.window_size = p
        # b batch_size   p : patch number f: feature

        if self.wa:
            y = self.activation(self.window_attention(self.avgpool(x).reshape(b, p)))#SE
            x = x * y.unsqueeze(2).repeat(1, 1, f)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        #print('qkv',qkv.shape)
        new_p = p // self.window_size
        # q, k, v = map(
        #     lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
        #                         h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        ##qkv矩阵
        q, k, v = map(
            lambda t: rearrange(t, 'b (new_w new_p) (head dim) -> b head new_p new_w dim',head=h, new_w=self.window_size), qkv)
        # q  {batch_size,head,patch_num,window_size,feature}
        start_time = time.time()
        if not self.prob:###不稀疏
            print('不稀疏')
            dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale#爱因斯坦求和

            if visual_feature_map:##特征图
                import matplotlib.pyplot as plt
                import seaborn as sns
                for i in range(int(dots.shape[1])):
                    sns.heatmap(torch.softmax(dots[0][i][0], dim=-1).cpu().detach().numpy())
                    # plt.savefig('heatmap_{0}.pdf'.format(i), format='pdf')
                    plt.show()

            if self.relative_pos_embedding:##相对位置bias
                dots += self.pos_embedding[self.relative_indices[:, :].to(torch.long)]
            else:
                dots += self.pos_embedding

            if self.shifted:
                dots[:, :, -new_p:] += self.left_mask

            attn = self.dropout(dots.softmax(dim=-1))
            out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        else:#应用稀疏策略
            print('稀疏')
            print('q',q.shape)
            print('k',k.shape)
            scores_top, index = self._compute_q_k(q, k)
            print('scores_top1', scores_top.shape)#[16, 12, 3, 25, 64]
            print('index1',index.shape)#([16, 12, 3, 25])
            scores_top = scores_top * self.scale
            print('scores_top2', scores_top.shape)#[16, 3, 12, 25, 64]
            print('pos_embedding[self.relative_indices[index].to(torch.long)]',self.pos_embedding[self.relative_indices[index].to(torch.long)].shape)#[16, 3, 12, 25, 64]
            print('self.relative_indices[index]',self.relative_indices[index].shape)#[16, 3, 12, 25, 64])
            if self.relative_pos_embedding:
                scores_top += self.pos_embedding[self.relative_indices[index].to(torch.long)]
                print('scores_top3', scores_top.shape)#[16, 12, 3, 25, 64])
            else:
                scores_top += self.pos_embedding

            print('scores_top4',scores_top.shape)
            #[16, 3, 12, 25, 64]
            #[16, 6, 6, 25, 64]
            #[16, 12, 3, 25, 64]
            if self.shifted:
                scores_top[:, :, -new_p:] += self.left_mask[index]
                print('scores_top5', scores_top.shape)
            context = self._get_initial_context(v, self.window_size)
            out = self._update_context(context, v, scores_top, index)
            print('out3',out.shape)#[16, 3, 12, 64, 32]
        out = rearrange(out, 'b head patch window_size dim -> b (patch window_size) (head dim)',
                        head=h)

        print('out1',out.shape)#16, 768, 96
        out = self.to_out(out)
        print('out2', out.shape)#16, 768, 96
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

    def _compute_q_k(self, q, k):#C
        B, Head, patch, L_Q, D = q.shape#batch_size,head,patch_num,window_size,feature
        print()
        _, _, _, L_K, _ = k.shape

        U_part = 5 * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = 5 * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)   #u是采样频率

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u)

        return scores_top, index

    def _prob_QK(self, Q, K, sample_k, n_top):##copy
        B, H, p, L_K, D = K.shape
        _, _, p, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, p, L_Q, L_K, D)  # B ,H ,P ,L_Q, L_K
        index_sample = torch.randint(L_K, (
            L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q             #从0~96中选出随机L_Q*sample_k个数
        K_sample = K_expand[:, :, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # 32 8 96 96 64  -> 32 8 96 25 64
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # 32 8 96 25  = 32 8 96 1 64  * 32 8 96 64 25
        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None, None],
                   torch.arange(H)[None, :, None, None],
                   torch.arange(p)[None, None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):###copy
        B, H, p, L_V, D = V.shape

        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, p, L_Q, V_sum.shape[-1]).clone()

        return contex

    def _update_context(self, context_in, V, scores, index):###copy
        B, H, P, L_V, D = V.shape

        attn = self.dropout(torch.softmax(scores, dim=-1))  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None, None],
        torch.arange(H)[None, :, None, None],
        torch.arange(P)[None, None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)

        return context_in


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted,
                 window_size, relative_pos_embedding, wa_dim, wa, prob, mask):
        super().__init__()

        ##注意力
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     wa_dim=wa_dim,
                                                                     wa=wa,
                                                                     prob=prob,
                                                                     mask=mask,
                                                                     )))
        ##
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

#Patch Merging层进行下采样。该模块的作用是做降采样，用于缩小分辨率，调整通道数 进而形成层次化的设计，同时也能节省一定运算量。
#patch Merging是一个类似于池化的操作，但是比Pooling操作复杂一些。池化会损失信息，patch Merging不会。
class PatchMerging(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, downscaling_factor=4):
        super(PatchMerging, self).__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor, out_channels)

    def forward(self, x):
        x = rearrange(x, 'b (p f) c -> b f (p c)', p=self.downscaling_factor)  # p c 是一个特征   f是代表分为了多少段
        x = self.linear(x)
        return x

#一个stage便可以看做由Patch Merging和Swin Transformer Block组成
class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads,
                 head_dim, window_size, wa, prob, mask,
                 relative_pos_embedding, wa_dim=4096):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        ##1
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)
        #
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([

                ##swin transformer
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          wa_dim=wa_dim, wa=wa, prob=prob, mask=mask),
                ##
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          wa_dim=wa_dim, wa=wa, prob=prob, mask=mask),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)##分割
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)

            ##
            x = shifted_block(x)
        # return x.permute(0, 3, 1, 2)
        return x


class DA_Net(nn.Module):
    def __init__(self, hidden_dim, layers, heads, channels, wa, prob, mask,
                 t=300, down_dim=1024, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        '''

        :param hidden_dim: 隐藏层个数
        :param layers: 每一层的block循环次数
        :param heads: head个数
        :param channels: 输入通道
        :param wa: bool 是否启用window attention
        :param prob: bool  是否启动prob attention
        :param mask: bool 是否启用mask
        :param t: 输入的time stamp 长度
        :param down_dim: 嵌入层的降维长度#
        :param num_classes: 输出类别
        :param head_dim: 每个head的维度
        :param window_size: 窗口大小
        :param downscaling_factors:每层下采样 倍数
        :param relative_pos_embedding: 是否使用相对位置信息
        '''
        super().__init__()
        self.downsample = nn.Linear(t, down_dim)#ts长度,3072
        print('down_dim',down_dim)#3072
        # self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim[0], layers=layers[0],
        #                           downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,
        #                           # embedding_dim=embedding_dim,
        #                           wa_dim = down_dim//downscaling_factors[0],wa=wa,prob=prob,mask=mask)
        # self.stage2 = StageModule(in_channels=hidden_dim[0], hidden_dimension=hidden_dim[1], layers=layers[1],
        #                           downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim, wa_dim = down_dim//downscaling_factors[0]//downscaling_factors[1],
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa=wa,prob=prob,mask=mask)
        # self.stage3 = StageModule(in_channels=hidden_dim[1], hidden_dimension=hidden_dim[2], layers=layers[2],
        #                           downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa_dim = down_dim//downscaling_factors[0]//downscaling_factors[1]//downscaling_factors[2],wa=wa,prob=prob,mask=mask)
        # self.stage4 = StageModule(in_channels=hidden_dim[2], hidden_dimension=hidden_dim[3], layers=layers[3],
        #                           downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
        #                           window_size=window_size, relative_pos_embedding=relative_pos_embedding,wa_dim = down_dim//downscaling_factors[0]//downscaling_factors[1]//downscaling_factors[2]//downscaling_factors[3],wa=wa,prob=prob,mask=mask)

        self.EncoderList = ModuleList()#
        for i in range(len(hidden_dim)):#hidden_dim=(96, 192, 62),
            layer = StageModule(#
                in_channels=channels if i == 0 else hidden_dim[i - 1],
                hidden_dimension=hidden_dim[i],
                layers=layers[i],
                downscaling_factor=downscaling_factors[i],
                num_heads=heads[i],
                head_dim=head_dim,
                window_size=window_size,
                relative_pos_embedding=relative_pos_embedding,
                wa_dim=down_dim // reduce(lambda x, y: x * y, downscaling_factors[:i + 1]),
                wa=wa,
                prob=prob,
                mask=mask
            )
            self.EncoderList.append(layer)###append
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim[-1]),##62,normalized_shape：归一化的维度，int（最后一维）list（list里面的维度）
            nn.Linear(hidden_dim[-1], num_classes)
        )

    def forward(self, ts):##ts为流入的训练数据
        print('ts',ts.shape)## torch.Size([16, 144, 62]),batchszie,
        ds = self.downsample(ts)  # B,C,L',（ts长度，3072）
        print('ds',ds.shape)#torch.Size([16, 144, 3072])
        x = ds.transpose(2, 1)
        print(x.shape)#torch.Size([16, 3072, 144])

        for Encoder in self.EncoderList:##
            x = Encoder(x)
        encoder = x
        print('encoder',encoder.shape)#([16, 192, 62])

        output = encoder.mean(dim=[1])#mean()函数的参数：dim=0,按列求平均值，返回的形状是（1，列数）；dim=1,
        # 按行求平均值，返回的形状是（行数，1）,默认不设置dim的时候，返回的是所有元素的平均值

        print('output',output.shape)#[16, 62]
        return ds, encoder, output, self.mlp_head(output)
        #embedding, encoder, output, pred_y = net(x)
