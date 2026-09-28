"""Standard and hierarchical TimePatch Transformer encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from TimeMAE.model.TimeMAE import Encoder
    from TimeMAE.model.layers import PositionalEmbedding
except ImportError as e:
    raise ImportError("TimeMAE must be available on PYTHONPATH.") from e

class TimeMAEPatchEmbed(nn.Module):
    def __init__(self,in_ch=3,patch_len=8,emb_dim=128,max_patches=5000):
        super().__init__(); self.patch_len=patch_len
        self.input_projection=nn.Conv1d(in_ch,emb_dim,patch_len,stride=patch_len)
        self.position=PositionalEmbedding(max_patches,emb_dim)
        class Args: pass
        a=Args(); a.d_model=emb_dim; a.attn_heads=4; a.layers=2
        a.dropout=.1; a.enable_res_parameter=1
        self.encoder=Encoder(a)
    def forward(self,x):
        T2=(x.shape[1]//self.patch_len)*self.patch_len
        x=self.input_projection(x[:,:T2].transpose(1,2)).transpose(1,2).contiguous()
        x=x+self.position(x)
        return self.encoder(x)

class PatchEmbed1D(nn.Module):
    def __init__(self,in_ch=3,patch_len=8,emb_dim=128):
        super().__init__(); self.patch_len=patch_len
        self.proj=nn.Linear(in_ch*patch_len,emb_dim)
    def forward(self,x):
        T2=(x.shape[1]//self.patch_len)*self.patch_len
        return self.proj(x[:,:T2].reshape(x.shape[0],T2//self.patch_len,-1))

class TimePatchTransformer(nn.Module):
    def __init__(self,in_ch=3,patch_len=8,emb_dim=128,depth=4,heads=4,
                 dropout=.1,num_classes=2,hierarchical=True):
        super().__init__()
        self.patch=TimeMAEPatchEmbed(in_ch,patch_len,emb_dim) if hierarchical else PatchEmbed1D(in_ch,patch_len,emb_dim)
        self.cls_token=nn.Parameter(torch.zeros(1,1,emb_dim))
        self.pos_emb=nn.Parameter(torch.zeros(1,512,emb_dim))
        layer=nn.TransformerEncoderLayer(emb_dim,heads,emb_dim*4,dropout,
                                         activation="gelu",batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,depth)
        self.norm=nn.LayerNorm(emb_dim)
        self.ssl_head=nn.Sequential(nn.Linear(emb_dim,emb_dim),nn.GELU(),nn.Linear(emb_dim,64))
        self.cls_head=nn.Linear(emb_dim,num_classes)
        nn.init.trunc_normal_(self.cls_token,std=.02); nn.init.trunc_normal_(self.pos_emb,std=.02)
    def forward_features(self,x):
        x=self.patch(x); B,N,_=x.shape
        x=torch.cat([self.cls_token.expand(B,-1,-1),x],1)
        return self.norm(self.encoder(x+self.pos_emb[:,:N+1]))[:,0]
    def forward_ssl(self,x): return F.normalize(self.ssl_head(self.forward_features(x)),dim=-1)
    def forward(self,x): return self.cls_head(self.forward_features(x))

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
