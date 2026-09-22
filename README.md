# Rolling our own Hadamard rotation for Prism's PQ2_0

Prism publishes the *runtime* that consumes Hadamard-rotated ternary weights and
the packing format, but **not** the offline tool that produces them
(see `volta-bonsai`). This is an attempt to write that tool.

**Result: the rotation half works. The quantisation half does not, yet.**

## What works

`rotate.py` folds a normalized Sylvester-Walsh Hadamard into the linear weights
and declares it as `prism.hadamard.*` metadata.

- Matrix mirrors the fork's own `ggml_gen_hadamard` (`src/llama-kv-cache.cpp:23`),
  seeded `1/sqrt(n)`. Verified self-inverse: `H@H == I` to **1.11e-16**.
- Normalisation `1/sqrt(block_size)` confirmed at `src/llama-model.cpp:2022`.
- `block_size = 512` — the largest power of two dividing all of Qwen3-4B's input
  dims (2560, 4096, 9728).
- 252 linear weights rotated (`attn_q/k/v/output`, `ffn_gate/up/down`).
  `token_embd` deliberately excluded to avoid the inverse-after-lookup case.
- `sign_mode = identity`, so no sign vectors are needed.

### Direction, derived not guessed

ggml index order is `[i=in, j=out]`. The runtime does:

```
x_rot = mul_mat(rot, x)        -> H . x          (H symmetric)
res   = mul_mat(w_new, x_rot)  -> w_new^T . H . x
```

We want `res == w_orig^T . x`, so `w_new = H . w_orig`. In numpy, where `A` is
`(n_out, n_in)`, that is `A_new = A_orig @ H` blockwise on the last axis.

### Proof it is correct

A rotation is mathematically neutral — the runtime cancels it exactly — so a
**rotated F16 model must behave like the unrotated one**. It does:

| model | output |
| :--- | :--- |
| `Qwen3-4B-F16` | "shorter wavelengths of light, like blue, are scattered more by the gas" |
| `Qwen3-4B-F16-rot512` | "shorter wavelengths of light, like blue, scatter more than the longer" |

Both coherent and correct. Not token-identical — F16 rounding of rotated weights
shifts a token here and there — but equivalent. Prompt eval drops 236.8 -> 132.6
t/s, the activation transform doing real work.

**The metadata also survives `llama-quantize`**: the PQ2_0 output still carries
all six `prism.hadamard.*` keys and the 252 weight names. That was an open risk
and it is closed.

## What does not work

Quantising the rotated model to PQ2_0 still produces garbage:

```
Long骋 Long创新型 fra LongrugceaMODEset倘若cea防范cea防范cea Long fix...
```

So **the rotation is necessary but not sufficient**.

## Hypothesis tested and falsified: it is not the signs

`rotate2.py` adds `sign_mode = explicit` with random +/-1 vectors, one per
distinct input width (2560, 4096, 9728 -> 16,384 values), folding `diag(s)`
in alongside `H`. Direction derived the same way: the runtime computes
`H . (s (*) x)`, so weights need `H . diag(s) . w`, i.e. `(A * s) @ H` in numpy.

**The signs are correct** - the F16 neutrality check passes again, coherent and
matching the original. But the ternary output is still garbage.

## Perplexity settles it

Measured on the V100, 20 chunks at `-c 512`, same corpus throughout:

| model | PPL |
| :--- | ---: |
| `Qwen3-4B-F16` (baseline) | **9.93** |
| `Qwen3-4B-PQ2_0` (no rotation) | 796,092 |
| `Qwen3-4B-rotPQ2_0` (rotation, identity signs) | 10,254,068 |
| `Qwen3-4B-rotsignPQ2_0` (rotation + random signs) | 6,314,600 |

A healthy model is ~10. Every ternary variant is 10^5 to 10^7 - noise, not
degradation. **Rotation makes it worse, not better.**

## Conclusion: the public PQ2_0 quantiser is a stub

If `quantize_row_pq2_0_ref` were merely unoptimised, the unrotated baseline
would be bad-but-functional and rotation would improve it. Neither holds.
Everything is catastrophically broken regardless of input, while Prism's own
released weights run correctly on this same runtime.

Supporting evidence from the source: `quantize_pq2_0` accepts a `quant_weights`
(imatrix) pointer and **ignores it** - both branches call the identical
`quantize_row_pq2_0_ref`:

```c
size_t quantize_pq2_0(..., const float * quant_weights) {
    if (!quant_weights) { quantize_row_pq2_0_ref(...); return ...; }
    for (row...)        { quantize_row_pq2_0_ref(...); }   // same call
}
```

So calibration cannot help either, even though `llama-imatrix` is built and
would run fine on this hardware. PQ2_0 is not in `tensor_requires_imatrix`, and
its kernel never reads the data.

**Prism published the packing format and the runtime, and withheld both things
that make the weights good: the rotation pipeline and the ternarisation.**

You cannot make your own Bonsai from what is public. This repo establishes that
with evidence rather than assumption - and the rotation half, which was the part
expected to be hard, works.

## What would change this

- PrismML answering [#242](https://github.com/PrismML-Eng/llama.cpp/issues/242)
  with the real pipeline.
- A ternary-native model (BitNet `TQ1_0`/`TQ2_0`), which needs no conversion at
  all - upstream llama.cpp and ik_llama both support those formats. Only small
  models exist so far.

## Files

| file | what |
| :--- | :--- |
| `rotate.py` | rotation only, `sign_mode = identity` |
| `rotate2.py` | adds random sign vectors, `sign_mode = explicit` |

Run with the venv at `~/bonsai/rotvenv` and
`PYTHONPATH=~/bonsai/llama.cpp/gguf-py`:

    rotate2.py in-F16.gguf out.gguf 512 signs
