#!/usr/bin/env python3
"""
minerva_compile.py v1.3.0-Athena
PTQ calibration fixes Bug 5 bias scale mismatch.
See inline docs for full details.
"""
import argparse, os, sys, struct, hashlib
from pathlib import Path
from typing import List, Tuple, Optional
try:
    import numpy as np
except ImportError:
    sys.exit("pip install numpy")

# ── ChaCha20 ──────────────────────────────────────────────────────────────────
def _r32(x,n): return ((x<<n)|(x>>(32-n)))&0xFFFFFFFF
def _cc20_block(key,nonce,ctr):
    C=[0x61707865,0x3320646e,0x79622d32,0x6b206574]
    K=list(struct.unpack_from('<8I',key)); N=list(struct.unpack_from('<3I',nonce))
    s=C+K+[ctr&0xFFFFFFFF]+N; x=s[:]
    for _ in range(10):
        def qr(a,b,c,d):
            x[a]=(x[a]+x[b])&0xFFFFFFFF;x[d]^=x[a];x[d]=_r32(x[d],16)
            x[c]=(x[c]+x[d])&0xFFFFFFFF;x[b]^=x[c];x[b]=_r32(x[b],12)
            x[a]=(x[a]+x[b])&0xFFFFFFFF;x[d]^=x[a];x[d]=_r32(x[d], 8)
            x[c]=(x[c]+x[d])&0xFFFFFFFF;x[b]^=x[c];x[b]=_r32(x[b], 7)
        qr(0,4,8,12);qr(1,5,9,13);qr(2,6,10,14);qr(3,7,11,15)
        qr(0,5,10,15);qr(1,6,11,12);qr(2,7,8,13);qr(3,4,9,14)
    return struct.pack('<16I',*[(x[i]+s[i])&0xFFFFFFFF for i in range(16)])

def cc20_encrypt(key,nonce,pt):
    out=bytearray()
    for i in range(0,len(pt),64):
        blk=_cc20_block(key,nonce,i//64); chunk=pt[i:i+64]
        out.extend(b^k for b,k in zip(chunk,blk))
    return bytes(out)

def b2s_mac(key,data):
    return hashlib.blake2s(data,key=key[:32],digest_size=32).digest()

# ── Quantization ──────────────────────────────────────────────────────────────
def qw(arr):
    """Quantize weight array to Q8 int8."""
    flat=arr.flatten().astype(np.float32)
    s=float(np.max(np.abs(flat)))
    if s==0: return bytes(len(flat))
    return np.clip(np.round(flat/s*127),-128,127).astype(np.int8).tobytes()

def q4(arr):
    flat=arr.flatten().astype(np.float32)
    s=float(np.max(np.abs(flat)))
    if s==0: return bytes((len(flat)+1)//2)
    q=np.clip(np.round(flat/s*7),-8,7).astype(np.int8)
    if len(q)%2: q=np.append(q,np.int8(0))
    packed=np.zeros(len(q)//2,dtype=np.uint8)
    for i in range(len(packed)):
        packed[i]=(int(q[2*i])&0xF)|((int(q[2*i+1])&0xF)<<4)
    return packed.tobytes()

def qbin(arr):
    flat=arr.flatten().astype(np.float32)
    bits=(flat>=0).astype(np.uint8)
    pad=(8-len(bits)%8)%8
    bits=np.concatenate([bits,np.zeros(pad,dtype=np.uint8)])
    return np.packbits(bits,bitorder='little').tobytes()

QFNS={'q8':qw,'q4':q4,'binary':qbin}
ACT_MAP={'relu':'MNV_ACT_RELU','sigmoid':'MNV_ACT_SIGMOID',
         'tanh':'MNV_ACT_TANH','linear':'MNV_ACT_LINEAR','sign':'MNV_ACT_SIGN'}

# ── Model ─────────────────────────────────────────────────────────────────────
class Layer:
    def __init__(self,W,b,act):
        self.weights=W.astype(np.float32); self.biases=b.astype(np.float32)
        self.activation=act
        self.in_size=W.shape[0]; self.out_size=W.shape[1] if W.ndim>1 else len(b)

class Model:
    def __init__(self,layers):
        self.layers=layers
        self.input_size=layers[0].in_size if layers else 0
        self.output_size=layers[-1].out_size if layers else 0

def load_npz(path):
    d=np.load(path,allow_pickle=True); layers=[]; i=0
    while f'layer_{i}_w' in d:
        W=d[f'layer_{i}_w'].astype(np.float32)
        b=d[f'layer_{i}_b'].astype(np.float32) if f'layer_{i}_b' in d else np.zeros(W.shape[1],dtype=np.float32)
        act=str(d[f'layer_{i}_act']) if f'layer_{i}_act' in d else 'relu'
        layers.append(Layer(W,b,act)); i+=1
    if not layers: sys.exit(f"No layers in {path}")
    return Model(layers)

# ── PTQ Calibration ───────────────────────────────────────────────────────────
def calibrate(model, X_calib, percentile=99.0):
    """
    Run calibration data through float model.
    Returns act_scales[i] = scale of layer i's OUTPUT activations.
    act_scales[0] = 127.0 (raw Q8 input range).

    Used to correctly scale biases for each layer:
        b_q = round(b * 127^2 / (128 * w_scale * act_scale_in))
    where act_scale_in is the scale of the INPUTS to that layer.
    """
    scales=[127.0]  # input scale = Q8 range
    a=X_calib.copy()
    for layer in model.layers:
        pre=a@layer.weights+layer.biases
        out=np.maximum(0,pre) if layer.activation.lower()=='relu' else pre
        s=float(np.percentile(np.abs(out),percentile))
        scales.append(max(s,1e-6))
        a=out
    print(f"  [PTQ] Layer input scales (index=layer receiving these activations):")
    for i,s in enumerate(scales):
        print(f"        scales[{i}] = {s:.4f}  ({'raw input' if i==0 else f'layer {i-1} output'})")
    return scales  # scales[i] is input scale for layer i

def qbias_ptq(bias, w_scale, act_scale_in):
    """
    PTQ bias quantization.

    The engine computes:
        acc = sum(W_q * x_q)   where W_q in [-127,127], x_q in [-127,127]
        acc_shifted = acc // 128

    W_q = W_float / w_scale * 127
    x_q = x_float / act_scale_in * 127

    So:
        acc = (127/w_scale) * (127/act_scale_in) * dot_float * N_inputs
        acc//128 = 127^2 / (128 * w_scale * act_scale_in) * dot_float

    For bias to add at the same scale:
        b_q = round(b_float * 127^2 / (128 * w_scale * act_scale_in))
    """
    if w_scale<1e-8 or act_scale_in<1e-8:
        return bytes(len(bias))
    factor = (127.0**2) / (128.0 * w_scale * act_scale_in)
    b_q = np.clip(np.round(bias*factor),-128,127).astype(np.int8)
    return b_q.tobytes()

# ── Compiler ──────────────────────────────────────────────────────────────────
class Compiler:
    def __init__(self,model,key,quant,target,act_scales=None):
        self.model=model; self.key=key; self.quant=quant
        self.target=target; self.act_scales=act_scales
        self.progmem='avr' in target.lower() or 'tiny' in target.lower() or 'mega' in target.lower()

    def _quant_layer(self,idx,layer):
        fn=QFNS[self.quant]
        w_bytes=fn(layer.weights.T)
        w_scale=float(np.max(np.abs(layer.weights)))
        if self.act_scales is not None:
            # PTQ: use calibrated input activation scale
            act_in=self.act_scales[idx]
            b_bytes=qbias_ptq(layer.biases,w_scale,act_in)
        else:
            # Independent (v1.2 fallback)
            bs=float(np.max(np.abs(layer.biases)))
            b_q=np.clip(np.round(layer.biases/bs*127),-128,127).astype(np.int8) if bs>0 \
                else np.zeros_like(layer.biases,dtype=np.int8)
            b_bytes=b_q.tobytes()
        return w_bytes,b_bytes

    def compile(self):
        blob=bytearray(); offsets=[]; debug={}
        for i,layer in enumerate(self.model.layers):
            wb,bb=self._quant_layer(i,layer)
            offsets.append((len(blob),len(wb),len(bb)))
            blob.extend(wb); blob.extend(bb)
        plaintext=bytes(blob)

        # Build debug arrays
        off=0
        for i,layer in enumerate(self.model.layers):
            wo,wl,bl=offsets[i]
            W_q=np.frombuffer(plaintext[off:off+wl],dtype=np.int8).reshape(layer.out_size,layer.in_size)
            b_q=np.frombuffer(plaintext[off+wl:off+wl+bl],dtype=np.int8)
            debug[f'W{i}T_q']=W_q; debug[f'b{i}_q']=b_q
            off+=wl+bl

        nonce=os.urandom(12)
        ct=cc20_encrypt(self.key,nonce,plaintext)
        mac=b2s_mac(self.key,ct)
        return self._emit_c(ct,nonce,mac,offsets), self._emit_h(ct,offsets), debug

    def calibrate_confidence(self,X_calib,debug):
        """10th percentile of max output logit on calibration data."""
        m=self.model; maxl=[]
        for sample in X_calib:
            a=np.clip(np.round(sample*127),-128,127).astype(np.int32)
            for i in range(len(m.layers)):
                W=debug[f'W{i}T_q'].astype(np.int32)
                b=debug[f'b{i}_q'].astype(np.int32)
                a=np.clip(W@a//128+b,-128,127)
                if m.layers[i].activation.lower()=='relu': a=np.maximum(0,a)
            maxl.append(int(np.max(a)))
        return max(0,int(np.percentile(maxl,10)))

    def _ca(self,name,data):
        pm=' PROGMEM' if self.progmem else ''
        lines=[f'const uint8_t {name}[]{pm} = {{']
        for i in range(0,len(data),16):
            lines.append('    '+', '.join(f'0x{b:02X}' for b in data[i:i+16])+',')
        lines.append('};'); return '\n'.join(lines)

    def _emit_c(self,ct,nonce,mac,offsets):
        pm='PROGMEM' if self.progmem else ''
        m=self.model
        mode='PTQ-calibrated' if self.act_scales else 'independent'
        L=['/**',' * weights.c — AUTO-GENERATED by minerva_compile.py v1.3.0',
           f' * Bias quant: {mode}','*/',
           '','#include "weights.h"','#include "minerva.h"','#include "secrets.h"']
        if self.progmem: L.append('#include <avr/pgmspace.h>')
        L+=['',self._ca('mnv_encrypted_weights',ct),'']
        L+=[f'const mnv_crypto_header_t mnv_crypto_hdr {pm} = {{',
            '    .iv  = {'+', '.join(f'0x{b:02X}' for b in nonce)+'},',
            '    .mac = {'+', '.join(f'0x{b:02X}' for b in mac)+'},',
            f'    .weight_count = {sum(o[1] for o in offsets)}U,',
            f'    .bias_count   = {sum(o[2] for o in offsets)}U,',
            '};','']
        L.append(f'const mnv_layer_desc_t mnv_layers[{len(m.layers)}] {pm} = {{')
        for i,layer in enumerate(m.layers):
            act=ACT_MAP.get(layer.activation.lower(),'MNV_ACT_RELU')
            L+=[f'    [{i}] = {{',f'        .input_size  = {layer.in_size}U,',
                f'        .output_size = {layer.out_size}U,',f'        .activation  = {act},',
                f'        .weights = NULL,',f'        .biases  = NULL,','    }},']
        L+=['};','',
            'const mnv_model_t mnv_model = {',
            '    .version           = MNV_ABI_VERSION,',
            f'    .num_layers        = {len(m.layers)}U,',
            '    .layers            = mnv_layers,',
            '    .crypto            = &mnv_crypto_hdr,',
            '    .key               = MNV_DEVICE_KEY,',
            '    .encrypted_weights = mnv_encrypted_weights,',
            '    .encrypted_len     = sizeof(mnv_encrypted_weights),',
            '};']
        return '\n'.join(L)+'\n'

    def _emit_h(self,ct,offsets):
        m=self.model
        L=['/** weights.h — AUTO-GENERATED by minerva_compile.py v1.3.0 */',
           '#ifndef MNV_WEIGHTS_H','#define MNV_WEIGHTS_H','#include "minerva.h"','',
           f'#define MNV_INPUT_SIZE    {m.input_size}U',
           f'#define MNV_NUM_LAYERS    {len(m.layers)}U']
        for i,l in enumerate(m.layers): L.append(f'#define MNV_LAYER_{i}_SIZE  {l.out_size}U')
        L+=[f'#define MNV_OUTPUT_SIZE   {m.output_size}U',
            f'#define MNV_ENCRYPTED_LEN {len(ct)}U','',
            'extern const mnv_model_t         mnv_model;',
            'extern const mnv_crypto_header_t mnv_crypto_hdr;',
            f'extern const mnv_layer_desc_t    mnv_layers[{len(m.layers)}];',
            'extern const uint8_t             mnv_encrypted_weights[];','',
            '#endif /* MNV_WEIGHTS_H */']
        return '\n'.join(L)+'\n'

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser(description='MINERVA Compiler v1.3.0-Athena')
    p.add_argument('model',nargs='?'); p.add_argument('--key')
    p.add_argument('--target',default='atmega328p',
                   choices=['atmega328p','atmega2560','attiny85','stm32f0','stm32f4','host'])
    p.add_argument('--quant',default='q8',choices=['q8','q4','binary'])
    p.add_argument('--calibrate',metavar='FILE',
                   help='Calibration .npz with key "X" shape [N, input_size]')
    p.add_argument('--out-dir',default='.')
    p.add_argument('--gen-key',metavar='FILE')
    p.add_argument('--gen-demo',metavar='FILE')
    p.add_argument('--dump-weights',action='store_true')
    args=p.parse_args()

    if args.gen_key:
        k=os.urandom(32)
        with open(args.gen_key,'wb') as f: f.write(k)
        print(f'[minerva] Key generated: {args.gen_key}\n          Hex: {k.hex()}')
        print(f'          KEEP SECRET. Never commit.'); return

    if args.gen_demo:
        _gen_demo(args.gen_demo); return

    if not args.model: p.error('model required')
    if not args.key:   p.error('--key required')
    with open(args.key,'rb') as f: key=f.read(32)

    path=Path(args.model)
    if not path.exists(): sys.exit(f'Not found: {path}')
    print(f'[minerva] Loading {path}')
    model=load_npz(str(path))
    for i,l in enumerate(model.layers):
        print(f'          Layer {i}: {l.in_size} -> {l.out_size} ({l.activation})')

    act_scales=None
    if args.calibrate:
        cp=Path(args.calibrate)
        if not cp.exists(): sys.exit(f'Calib not found: {cp}')
        cd=np.load(str(cp),allow_pickle=True)
        if 'X' not in cd: sys.exit('Calibration .npz must have key "X"')
        X_c=cd['X'].astype(np.float32)
        print(f'[minerva] PTQ calibration: {len(X_c)} samples')
        act_scales=calibrate(model,X_c)
    else:
        print(f'[minerva] No calibration -- independent bias quant (add --calibrate for better accuracy)')

    compiler=Compiler(model,key,args.quant,args.target,act_scales)
    c_src,h_src,debug=compiler.compile()

    if args.calibrate:
        X_c=np.load(str(Path(args.calibrate)),allow_pickle=True)['X'].astype(np.float32)
        thresh=compiler.calibrate_confidence(X_c,debug)
        print(f'[minerva] Calibrated MNV_MIN_CONFIDENCE: {thresh}')

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'weights.c').write_text(c_src)
    (out/'weights.h').write_text(h_src)
    if args.dump_weights:
        np.savez(str(out/'weights_debug.npz'),**debug)
        print(f'[minerva] Debug weights: {out}/weights_debug.npz (use //128 not >>7 in Python)')

    print(f'[minerva] Compiled. Bias: {"PTQ-calibrated" if act_scales else "independent"}')
    print(f'          weights.c + weights.h -> {out}')

def _gen_demo(out_path):
    np.random.seed(42); N,DIM=2000,8
    def mc(n,mean,std,label):
        return np.clip(np.random.randn(n,DIM)*std+mean,-1,1).astype(np.float32),np.full(n,label,dtype=np.int32)
    X=np.vstack([mc(N,.5,.15,0)[0],mc(N,-.5,.15,1)[0],mc(N,0,.55,2)[0],mc(N,0,.05,3)[0]])
    y=np.concatenate([mc(N,.5,.15,0)[1],mc(N,-.5,.15,1)[1],mc(N,0,.55,2)[1],mc(N,0,.05,3)[1]])
    idx=np.random.permutation(len(y)); X,y=X[idx],y[idx]
    sp=int(.8*len(y)); Xtr,ytr=X[:sp],y[:sp]
    Yte=np.zeros((len(y)-sp,4),dtype=np.float32); Yte[np.arange(len(y)-sp),y[sp:]]=1
    Ytr=np.zeros((sp,4),dtype=np.float32); Ytr[np.arange(sp),ytr]=1
    LAYERS=[DIM,16,8,4]; p={}
    for i in range(3):
        p[f'W{i}']=(np.random.randn(LAYERS[i],LAYERS[i+1])*np.sqrt(2/LAYERS[i])).astype(np.float32)
        p[f'b{i}']=np.zeros(LAYERS[i+1],dtype=np.float32)
    def relu(x): return np.maximum(0,x)
    def sm(x): e=np.exp(x-x.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
    for _ in range(100):
        idx2=np.random.permutation(sp)
        for s in range(0,sp,64):
            b=idx2[s:s+64]; Xb=Xtr[b]; Yb=Ytr[b]
            h0=relu(Xb@p['W0']+p['b0']); h1=relu(h0@p['W1']+p['b1']); pr=sm(h1@p['W2']+p['b2'])
            m2=len(Xb); dz=(pr-Yb)/m2
            p['W2']-=.01*(h1.T@dz); p['b2']-=.01*dz.sum(0); da=dz@p['W2'].T; dz=da*(h1>0)
            p['W1']-=.01*(h0.T@dz); p['b1']-=.01*dz.sum(0); da=dz@p['W1'].T; dz=da*(h0>0)
            p['W0']-=.01*(Xb.T@dz); p['b0']-=.01*dz.sum(0)
    h0=relu(X[sp:]@p['W0']+p['b0']); h1=relu(h0@p['W1']+p['b1']); pr=sm(h1@p['W2']+p['b2'])
    print(f'[minerva] Demo accuracy: {float(np.mean(np.argmax(pr,1)==y[sp:])):.3f}')
    save={f'layer_{i}_w':p[f'W{i}'] for i in range(3)}
    save.update({f'layer_{i}_b':p[f'b{i}'] for i in range(3)})
    save.update({f'layer_{i}_act':np.array('relu' if i<2 else 'linear') for i in range(3)})
    np.savez(out_path,**save)
    calib=str(Path(out_path).parent/'calib.npz')
    np.savez(calib,X=Xtr[:500])
    print(f'[minerva] Saved: {out_path}'); print(f'[minerva] Calib: {calib}')

if __name__=='__main__': main()
