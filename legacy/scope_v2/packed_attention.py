"""Two-pass split decode over actual persistent INT4 codes/scales. BF16, D128, g32 only.

No global dense K/V reconstruction. Scores and probabilities round to BF16 at
the original eager boundaries. Reduction order is different and must be gated.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _scores(Q,PK,SK,b,h,kv,t,d,N,SQB:tl.constexpr,SQH:tl.constexpr,
            SPB:tl.constexpr,SPH:tl.constexpr,SPT:tl.constexpr,
            SSB:tl.constexpr,SSH:tl.constexpr,SST:tl.constexpr,SCALE:tl.constexpr):
    q=tl.load(Q+b*SQB+h*SQH+d).to(tl.float32)
    u=tl.load(PK+b*SPB+kv*SPH+t[:,None]*SPT+d[None,:]//2,mask=t[:,None]<N,other=136).to(tl.int32)
    code=tl.where(d[None,:]%2==0,u&15,(u>>4)&15).to(tl.float32)-8.
    s=tl.load(SK+b*SSB+kv*SSH+t[:,None]*SST+d[None,:]//32,mask=t[:,None]<N,other=0.)
    k=(code*s).to(tl.bfloat16).to(tl.float32)
    raw=tl.sum(k*q[None,:],axis=1).to(tl.bfloat16).to(tl.float32)
    z=(raw*SCALE).to(tl.bfloat16).to(tl.float32)
    return tl.where(t<N,z,-float('inf'))

@triton.jit
def _normalizers(Q,PK,SK,M,L,N,H:tl.constexpr,REP:tl.constexpr,NS:tl.constexpr,
                 SQB:tl.constexpr,SQH:tl.constexpr,SPB:tl.constexpr,SPH:tl.constexpr,SPT:tl.constexpr,
                 SSB:tl.constexpr,SSH:tl.constexpr,SST:tl.constexpr,SCALE:tl.constexpr,TILE:tl.constexpr):
    bh=tl.program_id(0);b=bh//H;h=bh%H;split=tl.program_id(1)
    t=split*TILE+tl.arange(0,TILE);d=tl.arange(0,128)
    z=_scores(Q,PK,SK,b,h,h//REP,t,d,N,SQB,SQH,SPB,SPH,SPT,SSB,SSH,SST,SCALE)
    m=tl.max(z,0);l=tl.sum(tl.exp(z-m),0);offset=(b*H+h)*NS+split
    tl.store(M+offset,m);tl.store(L+offset,l)

@triton.jit
def _weighted_values(Q,PK,SK,PV,SV,M,L,Y,N,H:tl.constexpr,REP:tl.constexpr,NS:tl.constexpr,MAXS:tl.constexpr,
                     SQB:tl.constexpr,SQH:tl.constexpr,SPB:tl.constexpr,SPH:tl.constexpr,SPT:tl.constexpr,
                     SSB:tl.constexpr,SSH:tl.constexpr,SST:tl.constexpr,SCALE:tl.constexpr,TILE:tl.constexpr):
    bh=tl.program_id(0);b=bh//H;h=bh%H;split=tl.program_id(1);kv=h//REP
    splits=tl.arange(0,MAXS);off=(b*H+h)*NS+splits
    pm=tl.load(M+off,mask=splits<NS,other=-float('inf'));pl=tl.load(L+off,mask=splits<NS,other=0.)
    global_m=tl.max(pm,0);den=tl.sum(pl*tl.exp(pm-global_m),0)
    t=split*TILE+tl.arange(0,TILE);d=tl.arange(0,128)
    z=_scores(Q,PK,SK,b,h,kv,t,d,N,SQB,SQH,SPB,SPH,SPT,SSB,SSH,SST,SCALE)
    p=(tl.exp(z-global_m)/den).to(tl.bfloat16).to(tl.float32)
    u=tl.load(PV+b*SPB+kv*SPH+t[:,None]*SPT+d[None,:]//2,mask=t[:,None]<N,other=136).to(tl.int32)
    c=tl.where(d[None,:]%2==0,u&15,(u>>4)&15).to(tl.float32)-8.
    s=tl.load(SV+b*SSB+kv*SSH+t[:,None]*SST+d[None,:]//32,mask=t[:,None]<N,other=0.)
    v=(c*s).to(tl.bfloat16).to(tl.float32)
    y=tl.sum(p[:,None]*v,axis=0)
    tl.store(Y+((b*H+h)*NS+split)*128+d,y)

@triton.jit
def _finish(Y,OUT,H:tl.constexpr,NS:tl.constexpr,MAXS:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);s=tl.arange(0,MAXS);d=tl.arange(0,128)
    x=tl.load(Y+((b*H+h)*NS+s[:,None])*128+d[None,:],mask=s[:,None]<NS,other=0.)
    y=tl.sum(x,axis=0);tl.store(OUT+(b*H+h)*128+d,y)

def attention(q,layer,scale):
    if q.dtype!=torch.bfloat16 or q.shape[-2:]!=(1,128) or layer.bits!=4 or layer.group!=32 or not q.is_cuda:
        raise ValueError('UNSUPPORTED_PACKED_KERNEL_CONTRACT: CUDA BF16 one-query D128 INT4 g32 required')
    if layer.pk.stride()!=layer.pv.stride() or layer.sk.stride()!=layer.sv.stride():raise ValueError('packed K/V strides differ')
    q=q.contiguous();B,H=q.shape[:2];N=layer.get_seq_length();KH=layer.pk.shape[1]
    if H%KH:raise ValueError('invalid GQA grouping')
    ns=triton.cdiv(N,128);maxs=triton.next_power_of_2(ns)
    m=torch.empty((B,H,ns),device=q.device,dtype=torch.float32);l=torch.empty_like(m)
    partial=torch.empty((B,H,ns,128),device=q.device,dtype=torch.float32)
    out=torch.empty_like(q)
    strides=(q.stride(0),q.stride(1),layer.pk.stride(0),layer.pk.stride(1),layer.pk.stride(2),
             layer.sk.stride(0),layer.sk.stride(1),layer.sk.stride(2))
    _normalizers[(B*H,ns)](q,layer.pk,layer.sk,m,l,N,H,H//KH,ns,*strides,scale,128,num_warps=8)
    _weighted_values[(B*H,ns)](q,layer.pk,layer.sk,layer.pv,layer.sv,m,l,partial,N,H,H//KH,ns,maxs,*strides,scale,128,num_warps=8)
    _finish[(B,H)](partial,out,H,ns,maxs,num_warps=8)
    return out
