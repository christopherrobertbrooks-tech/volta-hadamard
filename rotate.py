#!/usr/bin/env python3
"""Fold a normalized Sylvester-Walsh Hadamard rotation into GGUF linear weights,
and declare it as prism.hadamard.* metadata so the PrismML runtime applies the
matching transform to activations.

Derivation of the direction (ggml index order [i=in, j=out]):
    runtime: x_rot = mul_mat(rot, x)      -> H . x      (H symmetric)
             res   = mul_mat(w_new, x_rot)-> w_new^T . H . x
    want   : res   = w_orig^T . x
    =>       w_new^T . H = w_orig^T  =>  w_new = H . w_orig      (H self-inverse)
    numpy A is (n_out, n_in) with A[j,i] = w[i,j], so A_new = A_orig @ H.

Usage: rotate.py <in.gguf> <out.gguf> <block_size>
"""
import sys, numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.constants import GGUFValueType

# Linear weights consumed by mul_mat / mul_mat_id.
# token_embd is deliberately excluded (it would need inverse-after-lookup).
#
# The *_exps tensors are MoE: all experts stacked into one 3-D tensor, ggml
# order (ne0=n_in, ne1=n_out, ne2=n_expert), so numpy sees (n_expert, n_out,
# n_in). The input dim is still the LAST numpy axis, which is the axis the
# blockwise rotation already operates on -- so the same code handles them once
# the names are listed and the rank check allows 3-D.
#
# ffn_gate_inp (the router) is deliberately NOT here. It feeds a discrete
# argmax over experts, so quantization error there flips which expert runs
# rather than perturbing a value. It stays unrotated and high precision.
ROTATE_SUFFIXES = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
)

# Anything matching this that we did NOT rotate gets reported, loudly. The
# earlier version silently skipped every MoE expert tensor -- 91% of gpt-oss-20b
# -- and still printed success.
LINEARISH = ("attn_q", "attn_k", "attn_v", "attn_output",
             "ffn_gate", "ffn_up", "ffn_down")
SKIP_OK = ("ffn_gate_inp.weight", "ffn_gate_inp.bias")   # router, on purpose

def hadamard(n):
    """Orthonormal Walsh-Hadamard, Sylvester recursion. Mirrors ggml_gen_hadamard.
    Self-inverse: H @ H == I."""
    H = np.zeros((n, n), dtype=np.float64)
    H[0, 0] = 1.0 / np.sqrt(n)
    s = 1
    while s < n:
        blk = H[:s, :s]
        H[s:2*s, :s] = blk
        H[:s, s:2*s] = blk
        H[s:2*s, s:2*s] = -blk
        s *= 2
    return H

def main():
    src, dst, block = sys.argv[1], sys.argv[2], int(sys.argv[3])
    assert block & (block - 1) == 0, "block_size must be a power of two"

    H = hadamard(block)
    err = np.abs(H @ H - np.eye(block)).max()
    assert err < 1e-9, f"H is not self-inverse (max err {err})"
    print(f"Hadamard {block}x{block} built; H@H==I to {err:.2e}")
    H32 = H.astype(np.float32)

    r = GGUFReader(src)
    arch = None
    for name, f in r.fields.items():
        if name == "general.architecture":
            arch = str(bytes(f.parts[f.data[0]]), "utf-8")
    assert arch, "no general.architecture"
    w = GGUFWriter(dst, arch)

    # --- copy every original KV verbatim -------------------------------------
    copied = 0
    for name, f in r.fields.items():
        if name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        t = f.types[0]
        try:
            if t == GGUFValueType.ARRAY:
                sub = f.types[1]
                if sub == GGUFValueType.STRING:
                    vals = [str(bytes(f.parts[i]), "utf-8") for i in f.data]
                else:
                    vals = [f.parts[i].tolist()[0] for i in f.data]
                w.add_key_value(name, vals, GGUFValueType.ARRAY, sub_type=sub)
            elif t == GGUFValueType.STRING:
                w.add_key_value(name, str(bytes(f.parts[f.data[0]]), "utf-8"), t)
            else:
                w.add_key_value(name, f.parts[f.data[0]].tolist()[0], t)
            copied += 1
        except Exception as e:
            print(f"  !! could not copy KV {name}: {e}")
    print(f"copied {copied} KV entries")

    # --- rotate the target tensors -------------------------------------------
    rotated_names, skipped = [], []
    for t in r.tensors:
        shp = tuple(int(x) for x in t.shape)          # ggml order: (ne0=in, ne1=out[, ne2=expert])
        data = t.data                                  # numpy, (..., n_out, n_in)
        if t.name.endswith(ROTATE_SUFFIXES) and len(shp) in (2, 3):
            n_in = shp[0]
            assert data.shape[-1] == n_in, f"{t.name}: expected last axis {n_in}, got {data.shape}"
            assert n_in % block == 0, f"{t.name}: n_in {n_in} not divisible by {block}"
            a = data.astype(np.float32)
            a = a.reshape(*a.shape[:-1], n_in // block, block)
            a = a @ H32                                # blockwise along the input dim
            a = a.reshape(*data.shape).astype(np.float16)
            w.add_tensor(t.name, a, raw_dtype=GGMLQuantizationType.F16)
            rotated_names.append(t.name)
        else:
            if (any(k in t.name for k in LINEARISH) and t.name.endswith(".weight")
                    and not t.name.endswith(SKIP_OK)
                    and "_norm" not in t.name and len(shp) >= 2):
                skipped.append((t.name, shp))
            w.add_tensor(t.name, data, raw_dtype=t.tensor_type)

    n3 = sum(1 for n in rotated_names if "_exps." in n or "_shexp." in n)
    print(f"rotated {len(rotated_names)} tensors ({n3} MoE expert), block={block}")
    if skipped:
        print(f"  !! {len(skipped)} linear weights NOT rotated -- check the suffix list:")
        for n, s in skipped[:8]:
            print(f"     {n}  shape={s}")
        raise SystemExit("refusing to write a partially rotated model")

    # --- declare the rotation -------------------------------------------------
    w.add_key_value("prism.hadamard.version", 1, GGUFValueType.UINT32)
    w.add_key_value("prism.hadamard.transform",
                    "normalized-sylvester-walsh-hadamard", GGUFValueType.STRING)
    w.add_key_value("prism.hadamard.axis", "input-last-dimension", GGUFValueType.STRING)
    w.add_key_value("prism.hadamard.block_size", block, GGUFValueType.UINT32)
    w.add_key_value("prism.hadamard.sign_mode", "identity", GGUFValueType.STRING)
    w.add_key_value("prism.hadamard.weight_names", rotated_names,
                    GGUFValueType.ARRAY, sub_type=GGUFValueType.STRING)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print("wrote", dst)

main()
