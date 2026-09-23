"""Dependency-free GGUF metadata reader.

Prints the keys that decide how much context a model can hold: the trained
context length and the attention geometry that sets KV-cache bytes per token.
Arrays are summarised (length, and for small int arrays the values) because
some hybrids store per-layer head_count_kv as an array, with 0 marking layers
that carry no KV cache at all (SSM / linear-attention layers).
"""
import struct, sys, json, os

T = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}

def rstr(f):
    n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode('utf-8', 'replace')

def rval(f, t):
    if t in T:
        fmt = T[t]; return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8: return rstr(f)
    if t == 9:
        et, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
        if et == 8:
            for _ in range(n): rstr(f)
            return f'<str[{n}]>'
        sz = struct.calcsize(T[et])
        if n > 512: f.seek(sz*n, 1); return f'<arr[{n}]>'
        return [struct.unpack(T[et], f.read(sz))[0] for _ in range(n)]
    raise ValueError(t)

WANT = ('context_length','block_count','head_count','head_count_kv','key_length','value_length',
        'embedding_length','sliding_window','full_attention_interval','attention.sliding_window',
        'kv_lora_rank','ssm.','rope.scaling','expert_count','expert_used_count')

def meta(path):
    out = {}
    with open(path,'rb') as f:
        assert f.read(4) == b'GGUF'
        ver, = struct.unpack('<I', f.read(4)); nt, nkv = struct.unpack('<QQ', f.read(16))
        for _ in range(nkv):
            k = rstr(f); t, = struct.unpack('<I', f.read(4)); v = rval(f, t)
            if k == 'general.architecture' or any(w in k for w in WANT): out[k] = v
    return out

for p in sys.argv[1:]:
    print(json.dumps({'file': os.path.basename(p), **meta(p)}))
