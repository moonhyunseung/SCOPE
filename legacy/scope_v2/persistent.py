"""Actual packed persistent KV layers, independent dense quantized reference and native adapters."""
import contextlib,dataclasses
import torch
from transformers.cache_utils import Cache,DynamicLayer
from scope.engineering import quantize,unpack

def encode(x,bits=4,group=32):
    lim=2**(bits-1)-1
    z=x.float().reshape(*x.shape[:-1],-1,group)
    scales=z.abs().amax(-1).clamp_min(1e-12)/lim
    codes=(z/scales.unsqueeze(-1)).round().clamp(-lim,lim).to(torch.int16).reshape(x.shape)
    if bits==4:
        u=(codes+8).to(torch.uint8);packed=u[...,::2]|(u[...,1::2]<<4)
    elif bits==8:packed=(codes+128).to(torch.uint8)
    else:raise ValueError('only 4/8 bits')
    return packed,scales

@dataclasses.dataclass
class Operand:
    layer:object
    key:bool
    def decode(self,start=0,end=None):
        p=self.layer.pk if self.key else self.layer.pv
        s=self.layer.sk if self.key else self.layer.sv
        return unpack(p[start:end,:,:self.layer.length],s[start:end,:,:self.layer.length],self.layer.bits,self.layer.group,self.layer.dtype)

class PackedLayer(DynamicLayer):
    def __init__(self,bits=4,group=32,mode='global',max_append=8,page=128):
        super().__init__();self.bits=bits;self.group=group;self.mode=mode;self.length=0;self.max_append=max_append;self.page=page;self.capacity=0
        self.pk=self.pv=self.sk=self.sv=None
    def allocate(self,k,batch,device):
        self.dtype=k.dtype;self.device=torch.device(device);H,D=k.shape[1],k.shape[-1]
        self.capacity=((k.shape[-2]+self.max_append+self.page-1)//self.page)*self.page
        self.pk=torch.empty((batch,H,self.capacity,D//2 if self.bits==4 else D),device=device,dtype=torch.uint8)
        self.pv=torch.empty_like(self.pk)
        self.sk=torch.empty((batch,H,self.capacity,D//self.group),device=device,dtype=torch.float32);self.sv=torch.empty_like(self.sk)
        self.is_initialized=True;self.keys=self.values=None
    def initialize(self,k,v,batch,device):
        self.allocate(k,batch,device)
        pk,sk=encode(k,self.bits,self.group);pv,sv=encode(v,self.bits,self.group)
        for name,x in [('pk',pk),('sk',sk),('pv',pv),('sv',sv)]:getattr(self,name)[...,:k.shape[-2],:].copy_(x.to(device).expand(batch,-1,-1,-1))
        self.length=k.shape[-2];self.is_initialized=True;self.keys=self.values=None
    def lazy_initialization(self,k,v):
        self.allocate(k,k.shape[0],k.device)
    def update(self,k,v,*args,**kwargs):
        if not self.is_initialized:self.lazy_initialization(k,v)
        pk,sk=encode(k,self.bits,self.group);pv,sv=encode(v,self.bits,self.group)
        end=self.length+k.shape[-2]
        if end>self.capacity:raise RuntimeError('declared packed capacity exceeded')
        for name,x in [('pk',pk),('sk',sk),('pv',pv),('sv',sv)]:getattr(self,name)[...,self.length:end,:].copy_(x)
        self.length+=k.shape[-2]
        # Only packed codes/scales persist; dense reconstruction is a temporary return value.
        if self.mode=='global':return Operand(self,True).decode(),Operand(self,False).decode()
        return Operand(self,True),Operand(self,False)
    def get_seq_length(self):return self.length
    def get_max_length(self):return self.capacity
    def crop(self,tokens_to_remove):
        n=max(0,self.length+tokens_to_remove) if tokens_to_remove<0 else min(self.length,tokens_to_remove)
        self.length=n
    def reset(self):
        self.pk=self.pv=self.sk=self.sv=None;self.length=0;self.is_initialized=False

class DenseStaticLayer(DynamicLayer):
    def __init__(self,max_append=8,page=128):super().__init__();self.max_append=max_append;self.page=page;self.length=0;self.capacity=0
    def initialize(self,k,v,batch,device):
        self.dtype=k.dtype;self.device=torch.device(device)
        self.capacity=((k.shape[-2]+self.max_append+self.page-1)//self.page)*self.page
        self.keys=torch.empty((batch,k.shape[1],self.capacity,k.shape[-1]),dtype=k.dtype,device=device);self.values=torch.empty_like(self.keys)
        self.length=k.shape[-2];self.keys[...,:self.length,:].copy_(k.to(device).expand(batch,-1,-1,-1));self.values[...,:self.length,:].copy_(v.to(device).expand(batch,-1,-1,-1));self.is_initialized=True
    def update(self,k,v,*args,**kwargs):
        end=self.length+k.shape[-2]
        if end>self.capacity:raise RuntimeError('declared dense capacity exceeded')
        self.keys[...,self.length:end,:].copy_(k);self.values[...,self.length:end,:].copy_(v);self.length=end
        return self.keys[...,:end,:],self.values[...,:end,:]
    def get_seq_length(self):return self.length
    def get_max_length(self):return self.capacity
    def crop(self,tokens_to_remove):self.length=max(0,self.length+tokens_to_remove) if tokens_to_remove<0 else min(self.length,tokens_to_remove)

class DenseQuantizedLayer(DynamicLayer):
    """Dense resident cache storing exactly the same dequantized representation."""
    def __init__(self,bits=4,group=32):super().__init__();self.bits=bits;self.group=group
    def initialize(self,k,v,batch,device):
        pk,sk=encode(k,self.bits,self.group);pv,sv=encode(v,self.bits,self.group)
        self.keys=unpack(pk,sk,self.bits,self.group,k.dtype).to(device).repeat(batch,1,1,1)
        self.values=unpack(pv,sv,self.bits,self.group,v.dtype).to(device).repeat(batch,1,1,1)
        self.dtype=k.dtype;self.device=torch.device(device);self.is_initialized=True
    def update(self,k,v,*args,**kwargs):
        pk,sk=encode(k,self.bits,self.group);pv,sv=encode(v,self.bits,self.group)
        return super().update(unpack(pk,sk,self.bits,self.group,k.dtype),unpack(pv,sv,self.bits,self.group,v.dtype),*args,**kwargs)

def build_cache(cpu_states,batch,kind,bits=4,group=32,device='cuda'):
    layers=[]
    for k,v in cpu_states:
        if kind=='dense':
            layer=DynamicLayer();layer.dtype=k.dtype;layer.device=torch.device(device);layer.is_initialized=True
            layer.keys=k.to(device).repeat(batch,1,1,1);layer.values=v.to(device).repeat(batch,1,1,1)
        elif kind=='dense_static':
            layer=DenseStaticLayer();layer.initialize(k,v,batch,device)
        elif kind=='dense_quant_reference':
            layer=DenseQuantizedLayer(bits,group);layer.initialize(k,v,batch,device)
        else:
            layer=PackedLayer(bits,group,kind);layer.initialize(k,v,batch,device)
        layers.append(layer)
    return Cache(layers=layers)

def payload(cache):
    seen=set();layers=[];total=0
    for index,layer in enumerate(cache.layers):
        row=dict(layer=index,valid_tokens=layer.get_seq_length(),fields={})
        for name in ['keys','values','pk','pv','sk','sv']:
            t=getattr(layer,name,None)
            if not isinstance(t,torch.Tensor):continue
            storage=t.untyped_storage();key=storage.data_ptr();n=storage.nbytes()
            cap=getattr(layer,'capacity',0)
            valid_bytes=int(t.numel()*t.element_size()*layer.get_seq_length()/cap) if cap else t.numel()*t.element_size()
            row['fields'][name]=dict(shape=list(t.shape),dtype=str(t.dtype),storage_bytes=n,valid_bytes=valid_bytes,padding_bytes=n-valid_bytes)
            if key not in seen:seen.add(key);total+=n
        layers.append(row)
    return dict(storage_bytes=total,layers=layers,unique_storages=len(seen),
        dense_persistent_fields=sum(isinstance(getattr(l,n,None),torch.Tensor) for l in cache.layers for n in ['keys','values']))

@contextlib.contextmanager
def execution(model,mode,chunk=64,validate_mask=False):
    import transformers.models.qwen2.modeling_qwen2 as module
    original=module.eager_attention_forward;stats=dict(calls=0,validation_rows=0)
    def observed(att,q,k,v,mask,*args,**kwargs):
        stats['calls']+=1
        if mode=='sdpa':
            import torch.nn.functional as F
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False,
                scale=kwargs.get('scaling',att.scaling),enable_gqa=q.shape[1]!=k.shape[1])
            return y.transpose(1,2).contiguous(),None
        if not isinstance(k,Operand):return original(att,q,k,v,mask,*args,**kwargs)
        if q.shape[-2]!=1:raise ValueError('packed native execution supports decode query length 1 only')
        if validate_mask and mask is not None and bool((mask!=0).any()):raise ValueError('unsupported nonzero/padded decode mask')
        if mode=='triton':
            from scope_v2.packed_attention import attention
            return attention(q,k.layer,att.scaling).transpose(1,2).contiguous(),None
        # Bounded workspace reference: original eager attention on fixed batch chunks.
        outputs=[]
        for start in range(0,q.shape[0],chunk):
            end=min(q.shape[0],start+chunk)
            m=mask[start:end] if mask is not None and mask.shape[0]>1 else mask
            y,_=original(att,q[start:end],k.decode(start,end),v.decode(start,end),m,*args,**kwargs);outputs.append(y)
        return torch.cat(outputs,0),None
    module.eager_attention_forward=observed
    try:yield stats
    finally:module.eager_attention_forward=original
