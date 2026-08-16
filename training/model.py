from __future__ import annotations
from dataclasses import dataclass
import math, torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size:int=16384; hidden_size:int=512; intermediate_size:int=1792
    num_hidden_layers:int=12; num_attention_heads:int=8; num_key_value_heads:int=2
    max_position_embeddings:int=2048; rope_theta:float=10000.0

class RMSNorm(nn.Module):
    def __init__(self, n): super().__init__(); self.weight=nn.Parameter(torch.ones(n))
    def forward(self,x): return F.rms_norm(x,(x.shape[-1],),self.weight,1e-6)

def rope(x, positions, theta):
    d=x.shape[-1]; inv=1/(theta**(torch.arange(0,d,2,device=x.device).float()/d)); a=positions.float()[:,None]*inv[None,:]
    c,s=a.cos()[None,None,:,:],a.sin()[None,None,:,:]; x1,x2=x[...,::2],x[...,1::2]
    return torch.stack((x1*c-x2*s,x1*s+x2*c),-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.h=c.num_attention_heads; self.kv=c.num_key_value_heads; self.d=c.hidden_size//self.h; self.theta=c.rope_theta
        self.q=nn.Linear(c.hidden_size,self.h*self.d,bias=False); self.k=nn.Linear(c.hidden_size,self.kv*self.d,bias=False); self.v=nn.Linear(c.hidden_size,self.kv*self.d,bias=False); self.o=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
    def forward(self,x):
        b,t,_=x.shape; pos=torch.arange(t,device=x.device); q=self.q(x).view(b,t,self.h,self.d).transpose(1,2); k=self.k(x).view(b,t,self.kv,self.d).transpose(1,2); v=self.v(x).view(b,t,self.kv,self.d).transpose(1,2)
        q,k=rope(q,pos,self.theta),rope(k,pos,self.theta); repeat=self.h//self.kv; k=k.repeat_interleave(repeat,1); v=v.repeat_interleave(repeat,1)
        return self.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(b,t,-1))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.n1=RMSNorm(c.hidden_size); self.attn=Attention(c); self.n2=RMSNorm(c.hidden_size); self.g=nn.Linear(c.hidden_size,c.intermediate_size,bias=False); self.u=nn.Linear(c.hidden_size,c.intermediate_size,bias=False); self.d=nn.Linear(c.intermediate_size,c.hidden_size,bias=False)
    def forward(self,x): x=x+self.attn(self.n1(x)); y=self.n2(x); return x+self.d(F.silu(self.g(y))*self.u(y))

class TinyLM(nn.Module):
    def __init__(self,c=ModelConfig()):
        super().__init__(); self.config=c; self.embed=nn.Embedding(c.vocab_size,c.hidden_size); self.blocks=nn.ModuleList([Block(c) for _ in range(c.num_hidden_layers)]); self.norm=RMSNorm(c.hidden_size)
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,mean=0,std=.02)
    def forward(self,input_ids,labels=None):
        x=self.embed(input_ids)
        for b in self.blocks: x=b(x)
        logits=F.linear(self.norm(x),self.embed.weight)
        loss=None if labels is None else F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1))
        return logits,loss
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

if __name__=="__main__":
    m=TinyLM(); print(f"parameters={m.parameter_count():,}"); assert 45_000_000 <= m.parameter_count() <= 55_000_000

