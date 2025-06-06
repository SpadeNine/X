# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from functools import partial
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

import torchvision.ops.roi_align as RoIAlign
from einops import rearrange


def generate_subimage_coordinates(H, W, h, w, num_windows):
    """
    生成子图的左上角和右下角坐标，并返回一个形状为 (n, 4) 的 PyTorch tensor。

    参数:
    H (int): 原始图像的高度
    W (int): 原始图像的宽度
    h (int): 子图的高度
    w (int): 子图的宽度

    返回:
    torch.Tensor: 形状为 (n, 4) 的张量，包含所有子图的左上角和右下角坐标
    """
    # assert H % h == 0 and W % w == 0, "H/h and W/w must be an integer"
    
    rows = int(round(H / h))
    cols = int(round(W / w))
    assert rows * cols == num_windows, f'H:{H}, W:{W}, h:{h}, w:{w}, rows:{H/h}, cols:{W/w}'
    coordinates = []
    for i in range(rows):
        for j in range(cols):
            x1 = j * w
            y1 = i * h
            x2 = x1 + w
            y2 = y1 + h
            coordinates.append([x1, y1, x2, y2])

    return torch.tensor(coordinates, dtype=torch.float32)


def slice_image_feature_fm9g(
    image_feature, num_windows=144, max_slice_nums=1000, num_ratio=1):
    # image_feature: b,c,h,w
    # num_queries of resampler. n
    # 
    bs = image_feature.shape[0]
    dtype, device = image_feature.dtype, image_feature.device
    feature_size = image_feature.shape[-2:]
    feature_height, feature_width = feature_size
    log_ratio = math.log(feature_width / feature_height)
    ratio = feature_height * feature_width / num_windows
    multiple = min(math.ceil(ratio), max_slice_nums)

    candidate_split_grids_nums = []
    for i in [multiple - 1, multiple, multiple + 1]:
        if i == 1 or i > max_slice_nums:
            continue
        candidate_split_grids_nums.append(i)

    candidate_grids = []
    # find best grid
    for split_grids_nums in candidate_split_grids_nums:
        m = 1
        while m <= split_grids_nums:
            if split_grids_nums % m == 0:
                candidate_grids.append([m, split_grids_nums // m])
            m += 1

    best_grid = [1, 1]
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error
    
    # (Iw * Ih) / n = Iw / Ih * h^2
    float_crop_height = math.sqrt(ratio / (feature_width / feature_height))
    float_crop_width = float_crop_height * (feature_width / feature_height)

    # print(float_crop_height, float_crop_width, feature_height, feature_width, )
    # print('true:', feature_height / float_crop_height, feature_width / float_crop_width)

    region_boxes = generate_subimage_coordinates(feature_height, feature_width, 
                                                float_crop_height, float_crop_width, num_windows)
    
    region_boxes = region_boxes.to(dtype=dtype, device=device).detach()
    batch_region_boxes = []
    for i in range(bs):
        batch_id = torch.ones_like(region_boxes)[:, :1] * i
        batch_region_boxes.append(torch.cat([batch_id, region_boxes], dim=1))
    batch_region_boxes = torch.cat(batch_region_boxes)

    return batch_region_boxes, best_grid, feature_width / feature_height

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    dtype = abs_pos.dtype

    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            num_queries,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            adaptive=False,
            max_size=(70, 70),
    ):
        super().__init__()
        self.grid_size = int(math.sqrt(num_queries))
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size
        self.kv_dim = kv_dim

        print("self.grid_size: ", self.grid_size)
        print("self.num_queries: ", self.num_queries)
        print("self.embed_dim: ", self.embed_dim)
        print("self.num_heads: ", self.num_heads)
        print("self.adaptive: ", self.adaptive)
        print("self.max_size: ", self.max_size)
        print("self.kv_dim: ", self.kv_dim)

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(self.kv_dim, self.grid_size)).float()
        )

        # four learnable expert embeddings
        self.feature_1x_embedding = nn.Parameter(torch.zeros(1,1, self.kv_dim))
        self.feature_4x_embedding = nn.Parameter(torch.zeros(1,1, self.kv_dim))

        # It is a 64 diverse embedding, not 
        self.query = nn.Parameter(torch.zeros(self.num_queries, self.kv_dim))
        trunc_normal_(self.query, std=.02)

        self.features_1x_projector = nn.Linear(in_features=self.kv_dim, out_features=self.kv_dim)
        self.features_4x_projector = nn.Linear(in_features=self.kv_dim, out_features=self.kv_dim)
                    
        self.attn = nn.MultiheadAttention(self.kv_dim, num_heads)
        self.ln_q = norm_layer(self.kv_dim)
        self.ln_kv = norm_layer(self.kv_dim)
        self.ln_proj = norm_layer(self.kv_dim)
        self.ln_post = norm_layer(self.kv_dim)
        self.out_mlp = nn.Sequential(nn.Linear(self.kv_dim, 2*self.kv_dim), nn.GELU(), nn.Linear(2*self.kv_dim, self.embed_dim))
        self.proj = nn.Parameter((self.embed_dim ** -0.5) * torch.randn(self.embed_dim, self.embed_dim))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cal_best_pooling_size(self, feature_wh_ratio=1.0):
        # candidate_pooling_sizes = [
        #     (4, 2), (3, 2), (4, 3), (3, 3), 
        #     (2, 4), (2, 3), (3, 4)
        # ] # w, h
        # log_feature_wh_ratio = math.log(feature_wh_ratio)
        best_pooling_size = (3, 3) # w, h
        # min_error = float("inf")
        # for candidate_pooling_size in candidate_pooling_sizes:
        #     w, h = candidate_pooling_size
        #     error = abs(log_feature_wh_ratio - math.log(w/h))
        #     if error < min_error:
        #         best_pooling_size = (h, w)
        #         min_error = error
        return best_pooling_size

    def adapt_unfold(self, input_embeds, spatial_size=(24, 24), best_grid=(1, 1), sampler_bins=1):
        # input_embeds: bs, n, c
        # spatial_size: feature map height, width
        # sampler_bins越大，采样点越多，细节越多
        input_embeds = input_embeds.permute(0, 3,1,2)

        resample_regions, best_grid, wh_ratio = slice_image_feature_fm9g(input_embeds, self.num_queries)

        output_size = self.cal_best_pooling_size(wh_ratio)
        aligned_feature = RoIAlign(input_embeds.float(), resample_regions.float(), output_size, 
                                    spatial_scale=1.0).to(dtype=input_embeds.dtype)
        unfold_input_embeds = aligned_feature.flatten(-2).permute(0, 2, 1)
        # bs*N, c, h, w -> bs*N,c,h*w -> bs*N, h*w, c
        return unfold_input_embeds

    def unfold(self, input_embeds, spatial_size=(24, 24), kernel_size=2, stride=2):
        # input_embeds: bs, n, c
        # spatial_size: feature map height, width
        input_embeds = input_embeds.permute(0, 2, 1).unflatten(-1, spatial_size)
        unfold_func = nn.Unfold(kernel_size=kernel_size, stride=stride)
        unfold_input_embeds = unfold_func(input_embeds) # bs, c* k**2, l
        unfold_input_embeds = unfold_input_embeds.unflatten(1, [-1, kernel_size ** 2]).permute(0, 3, 2, 1).flatten(0, 1)
        # bs, c*k**2, l -> bs, c, k**2, l -> bs, l, k**2, c -> bs*l, k**2, c
        return unfold_input_embeds
    
    def prepare_single_key_value(self, feature_1x, feature_2x, feature_4x, feature_8x, tgt_size=(24, 24), attn_mask=None, dtype=torch.bfloat16):
        """Prepare KV in a 4*9 manner"""
        muti_res_feat_keys = []
        muti_res_feat_values = []
        bs = 1

        _, _, height, width = feature_1x.shape
        feature_1x = feature_1x.to(torch.float32)
        feature_4x = F.interpolate(feature_1x, size=(height*4, width*4), mode='bilinear', align_corners=False)

        feature_list = [feature_1x, feature_4x]
        embedding_list =  [self.feature_1x_embedding, self.feature_4x_embedding]
        projector_list = [self.features_1x_projector, self.features_4x_projector]

        for feature, embedding, projector in zip(feature_list, embedding_list, projector_list):
            if feature is None:
                continue
            
            feature = feature.to(torch.bfloat16)
            feature = projector(feature.permute(0,2,3,1))
            
            key_height = feature.shape[1]
            key_width = feature.shape[2]
            key_pos_embed = get_abs_pos(self.pos_embed, (key_height, key_width)) #torch.Size([550, 4096])
            feature = rearrange(feature,'b h w c -> b (h w) c')  #torch.Size([1, 50, 44, 4096]) to torch.Size([1, 2200, 4096])
            feature = self.ln_kv(feature) #torch.Size([1, 2304, 4096]) #torch.Size([1, 9216, 4096])
            key = feature + key_pos_embed[None].to(dtype=dtype) + embedding.to(dtype=dtype) 
            value = feature
            key = key.reshape(bs, key_height, key_width, self.kv_dim) #torch.Size([1, 48, 48, 4096]) #torch.Size([1, 96, 96, 4096])
            key = self.adapt_unfold(key) #torch.Size([64, 9, 4096])  #torch.Size([64, 9, 4096])
            value = value.reshape(bs, key_height, key_width, self.kv_dim)
            value = self.adapt_unfold(value)# torch.Size([64, 9, 4096])  #torch.Size([64, 9, 4096])
            muti_res_feat_keys.append(key) #torch.Size([64, 9, 4096])  #torch.Size([64, 9, 4096])
            muti_res_feat_values.append(value)

        muti_res_feat_keys = torch.cat(muti_res_feat_keys, dim=1) # (64, 36, 5120)
        muti_res_feat_values = torch.cat(muti_res_feat_values, dim=1) # (64, 36, 5120)

        return muti_res_feat_keys, muti_res_feat_values

    def query_with_parallel_attn(self, key_list, value_list, dtype=torch.bfloat16):
        """Prepare Q and do attn"""
        bs = len(key_list) #7

        keys = torch.cat(key_list, dim=0)
        values = torch.cat(value_list, dim=0)

        attn_results = []
        for query_now in [self.query]:
            q = self.ln_q(query_now)
            query = self._repeat(q, bs) + self.pos_embed[None].to(dtype=dtype)
            query = self.unfold(query, spatial_size=(self.grid_size, self.grid_size), kernel_size=1, stride=1) #torch.Size([1008, 1, 4096])
            
            out, attn_weights = self.attn(      #[1, 1008, 4096]
                query.permute(1, 0, 2),         #torch.Size([1, 1008, 4096])
                keys.permute(1, 0, 2),   #torch.Size([36, 1008, 4096])
                values.permute(1, 0, 2)
            )
            # out->1, bs*l, c
            get = out[0].unflatten(0, [bs, -1]) # bs, l, c  #torch.Size([7, 64, 4096])
            get = self.ln_proj(get)
            attn_results.append(get)

        x = torch.cat(attn_results, dim=2)  #torch.Size([7, 64, 16384])
        x = self.ln_post(x)  #torch.Size([7, 64, 4096])
        x = self.out_mlp(x)
        x = x @ self.proj #torch.Size([7, 64, 4096])

        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(0).repeat(N, 1, 1)
    
    def partition_list(self, input_list, lengths):
        """
        按照指定的长度划分列表。

        参数:
        input_list (list): 要划分的原始列表。
        lengths (list): 一个包含划分长度的整数列表。

        返回:
        list: 一个包含子列表的列表，每个子列表的长度由 lengths 指定。
        """
        result = []
        current_index = 0
        for length in lengths:
            if current_index + length > len(input_list):
                raise ValueError("划分长度超过了列表的总长度")
            sublist = input_list[current_index:current_index + length]
            result.append(sublist)
            current_index += length
        if current_index != len(input_list):
            raise ValueError("划分长度和列表总长度不一致")
        return result
    

    def forward(self, features, patch_sizes):
        # achor2 = time.time() - start  #0.38
        # print(f'achor2: {achor2 - achor1}')

        patch_sizes = [(int(patch_sizes[i][0]), int(patch_sizes[i][1])) for i in range(patch_sizes.shape[0])]

        features_1x = [] #list torch.Size([1, 1024, 25, 22])
        for i in range(len(features)):
            h, w = patch_sizes[i]
            feature = features[i][:h*w, :].unsqueeze(0)
            feature = feature.permute(0, 2, 1)  #torch.Size([1, 1024, 25*22])
            feature = feature.unflatten(2, [h, w])  #torch.Size([1, 1024, 25, 22])
            features_1x.append(feature)

        feature_scale_mask =  7
        features_2x = []
        features_4x = []
        features_8x = []
        
        projected_image_features = []
        
        def get_element(features, index):
            if len(features) == 0:
                return None
            return features[index]
        
        key_list = []
        value_list = []
        for i in range(len(patch_sizes)):
            key, value = self.prepare_single_key_value(
                get_element(features_1x, i), 
                get_element(features_2x, i), 
                get_element(features_4x, i), 
                get_element(features_8x, i),
                get_element(patch_sizes, i)) # (64, 36, 5120)
            key_list.append(key)
            value_list.append(value)

        projected_image_features = self.query_with_parallel_attn(key_list, value_list)

        return projected_image_features


