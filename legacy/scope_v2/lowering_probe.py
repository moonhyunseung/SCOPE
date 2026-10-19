"""Read-only diagnostic clone of the existing weighted-values kernel.

No optimized execution family. Only this clone stores Z/P. Its Y must be
bitwise equal to the unchanged production kernel on each observed operand.
Even that control does not prove compiler-internal instruction identity.
"""
import torch
import triton
import triton.language as tl
from scope_v2.packed_attention import _scores,_normalizers,_finish,attention

@triton.jit
def _debug_values(Q,PK,SK,PV,SV,M,L,Y,ZOUT,POUT,N,H:tl.constexpr,REP:tl.constexpr,NS:tl.constexpr,MAXS:tl.constexpr,
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
    tl.store(ZOUT+bh*N+t,z,mask=t<N)
    tl.store(POUT+bh*N+t,p,mask=t<N)

def observe(q,layer,scale):
    q=q.contiguous();B,H=q.shape[:2];N=layer.get_seq_length();KH=layer.pk.shape[1]
    ns=triton.cdiv(N,128);maxs=triton.next_power_of_2(ns)
    m=torch.empty((B,H,ns),device=q.device,dtype=torch.float32);l=torch.empty_like(m)
    partial=torch.empty((B,H,ns,128),device=q.device,dtype=torch.float32);out=torch.empty_like(q)
    z=torch.empty((B,H,1,N),device=q.device,dtype=torch.float32);p=torch.empty_like(z)
    strides=(q.stride(0),q.stride(1),layer.pk.stride(0),layer.pk.stride(1),layer.pk.stride(2),layer.sk.stride(0),layer.sk.stride(1),layer.sk.stride(2))
    _normalizers[(B*H,ns)](q,layer.pk,layer.sk,m,l,N,H,H//KH,ns,*strides,scale,128,num_warps=8)
    _debug_values[(B*H,ns)](q,layer.pk,layer.sk,layer.pv,layer.sv,m,l,partial,z,p,N,H,H//KH,ns,maxs,*strides,scale,128,num_warps=8)
    _finish[(B,H)](partial,out,H,ns,maxs,num_warps=8)
    original=attention(q,layer,scale)
    return out,z,p,bool(torch.equal(out,original)),original
