from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from posetimation.layers import CHAIN_RSB_BLOCKS,ChainOfBasicBlocks
from posetimation.zoo.backbones.vit import *
from posetimation.zoo.deformable_attention.deformable_attention_2d import *
from posetimation.zoo.layers.corss_block import *


class Deformable_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None , map_size = (16,12)
                 ):
        super().__init__()

        self.map_h = map_size[0]
        self.map_w = map_size[1]

        self.norm1 = norm_layer(dim)
        self.deformable_attn =  DeformableAttention2D(
            dim = dim,                   # feature dimensions
            dim_head = dim//num_heads,               # dimension per head
            heads = num_heads,                   # attention heads
            dropout = 0.,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = None,        # number of offset groups, should be multiple of heads
            offset_kernel_size = 6,      # offset kernel size
        )


        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x ):
        b,n,d = x.shape

        x = x + self.drop_path( self.deformable_attn( self.norm1(x).permute(0, 2,1).reshape(b,d,self.map_h,self.map_w) ).reshape(b,d,n).permute(0, 2,1)  )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class Deformable_Cross_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None , map_size = (16,12),block_num=1,fusion=False
                 ):
        super().__init__()

        self.map_h = map_size[0]
        self.map_w = map_size[1]
        self.block_num = block_num
        self.fusion = fusion

        if self.block_num == 1:

            self.y_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            )

            self.norm1 = norm_layer(dim)
            self.deformable_attn =  DeformableCrossAttention2D(
                dim = dim,
                feat_dim  = dim ,
                dim_head = dim//num_heads,               # dimension per head
                heads = num_heads,                   # attention heads
                dropout = 0.,                # dropout
                downsample_factor = 4,       # downsample factor (r in paper)
                offset_scale = 4,            # scale of offset, maximum offset
                offset_groups = None,        # number of offset groups, should be multiple of heads
                offset_kernel_size = 6,      # offset kernel size
            )
        elif self.block_num == 2:
            if self.fusion:
                self.y_proj = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim)
                )

                self.norm1 = norm_layer(dim)
                self.deformable_attn = DeformableCrossAttention2D(
                    dim=dim,
                    feat_dim=dim,
                    dim_head=dim // num_heads,  # dimension per head
                    heads=num_heads,  # attention heads
                    dropout=0.,  # dropout
                    downsample_factor=4,  # downsample factor (r in paper)
                    offset_scale=4,  # scale of offset, maximum offset
                    offset_groups=None,  # number of offset groups, should be multiple of heads
                    offset_kernel_size=6,  # offset kernel size
                )

                self.norm = norm_layer(dim)
                self.cross_attn = CrossAttention(dim, dim , heads=num_heads)
            else:
                self.y_proj = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim)
                )

                self.norm1 = norm_layer(dim)
                self.deformable_attn = DeformableAttention2D(
                    dim=dim,
                    dim_head=dim // num_heads,  # dimension per head
                    heads=num_heads,  # attention heads
                    dropout=0.,  # dropout
                    downsample_factor=4,  # downsample factor (r in paper)
                    offset_scale=4,  # scale of offset, maximum offset
                    offset_groups=None,  # number of offset groups, should be multiple of heads
                    offset_kernel_size=6,  # offset kernel size
                )
                self.norm = norm_layer(dim)
                self.cross_attn = CrossAttention(dim, dim , heads=num_heads)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x , y ):
        b,n,d = x.shape
        if self.block_num == 1:
            y = self.y_proj(y)

            x = x + self.drop_path( self.deformable_attn(
                self.norm1(x).permute(0, 2,1).reshape(b,d,self.map_h,self.map_w) ,y.permute(0, 2,1).reshape(b,d,self.map_h,self.map_w) ).reshape(b,d,n).permute(0, 2,1)  )
        else:
            if self.fusion:
                y = self.y_proj(y)
                x = x + self.drop_path(self.deformable_attn(
                    self.norm1(x).permute(0, 2, 1).reshape(b, d, self.map_h, self.map_w),
                    y.permute(0, 2, 1).reshape(b, d, self.map_h, self.map_w)).reshape(b, d, n).permute(0, 2, 1))
                x = x + self.drop_path(self.cross_attn(self.norm(x), y , y ))

            else:
                y = self.y_proj(y)

                x = x + self.drop_path(self.deformable_attn( self.norm1(x).permute(0, 2,1).reshape(b,d,self.map_h,self.map_w) ).reshape(b,d,n).permute(0, 2,1)  )
                x = x + self.drop_path(self.cross_attn(self.norm(x), y, y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class Bidirectional_Deformable_Cross_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None , map_size = (16,12),
                 ):
        super().__init__()

        self.map_h = map_size[0]
        self.map_w = map_size[1]



        self.y_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

        self.norm_forward = norm_layer(dim)
        self.norm_backward = norm_layer(dim)

        self.deformable_attn_forward = DeformableCrossAttention2D(
            dim=dim,
            feat_dim=dim,
            dim_head=dim // num_heads,  # dimension per head
            heads=num_heads,  # attention heads
            dropout=0.,  # dropout
            downsample_factor=4,  # downsample factor (r in paper)
            offset_scale=4,  # scale of offset, maximum offset
            offset_groups=None,  # number of offset groups, should be multiple of heads
            offset_kernel_size=6,  # offset kernel size
        )
        self.deformable_attn_backward = DeformableCrossAttention2D(
            dim=dim,
            feat_dim=dim,
            dim_head=dim // num_heads,  # dimension per head
            heads=num_heads,  # attention heads
            dropout=0.,  # dropout
            downsample_factor=4,  # downsample factor (r in paper)
            offset_scale=4,  # scale of offset, maximum offset
            offset_groups=None,  # number of offset groups, should be multiple of heads
            offset_kernel_size=6,  # offset kernel size
        )

        self.norm_cross_forward = norm_layer(dim)
        self.norm_cross_backward = norm_layer(dim)

        self.cross_attn_forward = CrossAttention(dim, dim , heads=num_heads)
        self.cross_attn_backward = CrossAttention(dim, dim , heads=num_heads)


        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_mlp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x , y ):
        #x:(B,192*2,1024)
        b,n_2x,d = x.shape
        n = n_2x // 2

        y = self.y_proj(y)
        #split
        x_f , x_b = x.split(n, dim=1)
        # first, forward stream
        x_f = x_f + self.drop_path(self.deformable_attn_forward(
            self.norm_forward(x_f).permute(0, 2, 1).reshape(b, d, self.map_h, self.map_w),
            y.permute(0, 2, 1).reshape(b, d, self.map_h, self.map_w)).reshape(b, d, n).permute(0, 2, 1))
        x_f = x_f + self.drop_path(self.cross_attn_forward(self.norm_cross_forward(x_f), y , y ))

        # second, backward stream
        x_b = x_b + self.drop_path(self.deformable_attn_backward(
            self.norm_backward(x_b).permute(0, 2, 1).reshape(b, d, self.map_h, self.map_w),
            y.permute(0, 2, 1).reshape(b, d, self.map_h, self.map_w)).reshape(b, d, n).permute(0, 2, 1))
        x_b = x_b + self.drop_path(self.cross_attn_backward(self.norm_cross_backward(x_b), y , y ))
        #concat
        x = torch.cat([x_f, x_b], dim=1)


        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x

if __name__ == '__main__':
    # # model = Deformable_Block(1024, 16, mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., drop_path=0.,  map_size = (16,12) )
    # # model = Deformable_Cross_Block(1024, 16, mlp_ratio=4.,   map_size = (16,12) , block_num=2,fusion=True )
    # model = Bidirectional_Deformable_Cross_Block(1024, 16, mlp_ratio=4.,   map_size = (16,12) )
    # print("=> Number of parameters in the model: {}".format(sum(p.numel() for p in model.parameters())))
    #
    # x = torch.randn(1, 192*2,1024)
    # # x = torch.randn(1, 192,1024)
    # feat = torch.randn(1, 192,1024)
    #
    # y = model(x,feat)
    #
    # print(y.shape)
    #
    # # 假设 model 是你的模型实例
    #
    # param_count = {}
    # for name, module in model.named_modules():
    #     # 收集模块的参数
    #     params = list(module.parameters())
    #     # 累加参数量
    #     param_count[name] = sum(p.numel() for p in params)
    #
    # # 打印每个模块的参数量
    # for name, count in param_count.items():
    #     print(f"{name}: {count}")



    dpr = [x.item() for x in torch.linspace(0, 0.18, 4)]
    print(dpr)
    print("!!!")
