
from functools import partial
from timm.models.layers import DropPath
from ..backbones.vit import Block , trunc_normal_
from ..layers.rope import *


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 1000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,):
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)






class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,use_rotary=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.use_rotary=use_rotary

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_rotary:
            self.freqs_cis = precompute_freqs_cis(dim, 192 * 2)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.use_rotary:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#------from DSTA-------add group-----
class new_Spatiotemporal_Atten(nn.Module):

    def __init__(self, length=192,channel=1024,frame_num=3,multi_frame_fusion = False , group = 1 , head = 8 , mlp_ratio = 2 ,depth =2):
        super().__init__()

        self.frame_num = frame_num
        self.multi_frame_fusion = multi_frame_fusion
        self.group = group
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, length, channel))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.frame_num * group, channel))
        dpr = [x.item() for x in torch.linspace(0, 0.06, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=1024, num_heads=head, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(channel)

        ### temporal block
        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=1024, num_heads=head, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(channel)
        if self.multi_frame_fusion:
            self.project_fusion = nn.Sequential(
                nn.Linear(length * self.frame_num , length),
                nn.Linear(length, length),
            )
            self.hf_norm = nn.LayerNorm(length)
            self.final_norm = nn.LayerNorm(channel)
        self.project_primary = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.Linear(192, 192),
        )

    def forward(self, x):
        batch_size , n,d = x.shape[0] // self.frame_num, x.shape[1], x.shape[2]
        #spatial
        feat_S = x + self.Spatial_pos_embed
        for blk in self.Spatial_blocks:
            feat_S = blk(feat_S)
        feat_S = self.Spatial_norm(feat_S)

        #temporal
        tokens_temporal = torch.stack(x.split(batch_size, dim=0), dim=0)
        num_f , b , n ,d = tokens_temporal.shape
        # feat_T = tokens_temporal.reshape(b * n/self.group, num_f*self.group, d)
        feat_T = rearrange(tokens_temporal, 'f b (p g) c -> (b p) (f g) c' , g = self.group,b=b )
        feat_T = feat_T + self.Temporal_pos_embed
        for blk in self.Temporal_blocks:
            feat_T = blk(feat_T)
        feat_T = self.Temporal_norm(feat_T)
        feat_T = rearrange(feat_T, '(b p) (f g) c -> (b f) (p g) c', g=self.group,f = self.frame_num,b = b)

        #spatial-temporal agg
        feat_agg = torch.cat((feat_T, feat_S), dim=1)
        feat_agg = self.project_primary(feat_agg.permute(0, 2, 1))
        feat_agg = feat_agg.permute(0, 2, 1)
        if self.multi_frame_fusion:
            #multi-frame fusion
            mask_feature = torch.cat(torch.chunk(feat_agg, chunks=self.frame_num, dim=0), dim=1)
            feature_heatmap = self.final_norm(mask_feature)
            heatmap_feature = self.hf_norm(self.project_fusion(feature_heatmap.permute(0, 2, 1)))
            out_feature = heatmap_feature.permute(0, 2, 1)
            return out_feature
        else:
            return feat_agg


class STTrans(nn.Module):
    def __init__(
        self, h_dim, depth=2, num_heads=16, mlp_ratio=4,
          dropout=0., embed_dropout=0. ,frame_num = 3, order_type= 's_t' , fusion = False , use_pos_emb = False
    ):
        super().__init__()

        self.transformer_s = []
        self.transformer_t = []
        self.depth = depth
        self.frame_num = frame_num
        self.order_type = order_type
        self.use_pos_emb = use_pos_emb
        self.multi_frame_fusion = fusion


        dpr = [x.item() for x in torch.linspace(0, 0.10, 4)]
        for i in range(depth):
            self.transformer_s.append(Block(
                    dim=h_dim, num_heads=num_heads, mlp_ratio= mlp_ratio , qkv_bias=True, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=dpr[i],
                ))
            self.transformer_t.append(Block(
                    dim=h_dim, num_heads=num_heads, mlp_ratio= mlp_ratio , qkv_bias=True, qk_scale=None,
                    drop=0., attn_drop=0., drop_path=dpr[i],
                ))
        self.transformer_s = nn.ModuleList(self.transformer_s)
        self.transformer_t = nn.ModuleList(self.transformer_t)


        if self.use_pos_emb:
            self.spatial_pos_emb_list ,self.temporal_pos_emb_list = [] ,[]
            for i in range(depth):
                self.spatial_pos_emb = nn.Parameter(torch.zeros(1, 192, h_dim))
                self.temporal_pos_emb = nn.Parameter(torch.zeros(1, self.frame_num, h_dim))
                trunc_normal_(self.spatial_pos_emb, std=.02)
                trunc_normal_(self.temporal_pos_emb, std=.02)
                self.spatial_pos_emb_list.append(self.spatial_pos_emb)
                self.temporal_pos_emb_list.append(self.temporal_pos_emb)


        if self.multi_frame_fusion:
            self.project_fusion = nn.Sequential(
                nn.Linear(192 * self.frame_num, 192),
                nn.Linear(192, 192),
                nn.LayerNorm(192)
            )
            self.final_norm = nn.LayerNorm(h_dim)

    def forward(self, x):
        # x进来就是(B*self.frame_num,192,1024)
        B , N ,D= x.shape[0]//self.frame_num , x.shape[1] , x.shape[2]
        out = []

        if self.order_type=='s_t':
            for i in range(self.depth):
                if self.use_pos_emb:
                    x = x + self.spatial_pos_emb_list[i].cuda()
                x = self.transformer_s[i](x)
                x = x.reshape(B,self.frame_num,N,D).permute(0,2,1,3).reshape(B*N,self.frame_num,D)
                if self.use_pos_emb:
                    x = x + self.temporal_pos_emb_list[i].cuda()
                x = self.transformer_t[i](x)
                x = x.view(B,N,self.frame_num,D).permute(0,2,1,3).reshape(B*self.frame_num,N,D)
                out.append(x)
        elif self.order_type=='t_s':
            for i in range(self.depth):

                x = x.reshape(B,self.frame_num,N,D).permute(0,2,1,3).reshape(B*N,self.frame_num,D)
                if self.use_pos_emb:
                    x = x + self.temporal_pos_emb_list[i].cuda()
                x = self.transformer_t[i](x)
                x = x.view(B,N,self.frame_num,D).permute(0,2,1,3).reshape(B*self.frame_num,N,D)
                if self.use_pos_emb:
                    x = x + self.spatial_pos_emb_list[i].cuda()
                x = self.transformer_s[i](x)
                out.append(x)

        if self.multi_frame_fusion:
            # multi-frame fusion
            feat_agg = out[-1]
            feature = torch.cat(torch.chunk(feat_agg, chunks=self.frame_num, dim=0), dim=1)
            feature = self.final_norm(feature)
            feature = self.project_fusion(feature.permute(0, 2, 1))
            out_feature = feature.permute(0, 2, 1)
            return out_feature
        else:
            return out[-1]

if __name__ == '__main__':
    attn = new_Spatiotemporal_Atten(length=192,channel=1024,num_sup=2)
    batch_size = 1
    num_sup = 2
    x = torch.randn(batch_size*(num_sup+1), 192,1024)

    y = attn(x)

    print(y.shape)

    dpr = [x.item() for x in torch.linspace(0, 0.06, 2)]
    print(dpr)
    print("!!!")