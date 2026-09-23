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

---

# The tool turned out to be useful for something else

Prism's PQ2_0 is a dead end (their quantiser is a stub). But the rotation is
general: it should make weights easier for *any* aggressive quantiser. So:
quantise the same model from a plain F16 and from a rotated F16, and compare.

`llama-perplexity`, 20 chunks at `-c 512`, same corpus, run on the PrismML fork
(which applies the activation transform). Block 512, `sign_mode = identity`.

| quant | Qwen3-4B plain | rot | Qwen3-8B plain | rot |
| :--- | ---: | ---: | ---: | ---: |
| Q4_K_M | 10.0389 | 10.3035 *(+2.6%)* | 8.9124 | **8.6749** *(-2.7%)* |
| Q3_K_M | 11.2520 | **10.6494** *(-5.4%)* | 9.0948 | 9.1697 *(+0.8%)* |
| Q2_K | 15.4422 | **14.7720** *(-4.3%)* | 11.8696 | **11.3152** *(-4.7%)* |

## What survives two models

**Only Q2_K is consistent**: -4.3% and -4.7%, both directions agreeing. That is
the finding.

**Q4_K_M and Q3_K_M flip sign between models**, so those are noise at this
magnitude. An earlier version of this file claimed a clean crossover -- rotation
hurting at 4-bit and helping below -- derived from the 4B alone. The 8B killed
it. That claim was overfitting to one model and is retracted.

The surviving statement is narrower and better supported: **rotation improves
Q2_K by ~4-5% and does nothing reliable above 2 bits.** Still consistent with
the outlier argument, but at a coarser grain than one model suggested.

## MoE: the tool skipped 91% of the model

Expert weights are 3-D (`ffn_{gate,up,down}_exps.weight`, ggml order
`ne0=n_in, ne1=n_out, ne2=n_expert`) and matched neither the dense suffix list
nor the `len(shape) == 2` guard. On gpt-oss-20b that is **91.37% of all
parameters silently skipped** against 3.05% actually rotated -- and the tool
printed success. Fixed, and it now aborts rather than write a partial rotation.

The runtime side already worked. `build_lora_mm_id` (the MoE expert path,
`llama-graph.cpp:1605`) carries the same Hadamard hook as `build_lora_mm`, and
`llama_verify_hadamard_graph` already covers `GGML_OP_MUL_MAT_ID`. Only the
exporter was missing. The fork also keeps an architecture allowlist and refuses
to load folded weights for anything unverified; `olmoe` is not on it, so this
run needed a one-line local addition (`olmoe.cpp` issues no raw `ggml_mul_mat`,
so it meets the gate's condition -- not upstreamed).

**The round-trip is exact on MoE.** Rotating 48 expert tensors and inverting at
runtime returns the original model:

| OLMoE-1B-7B | PPL |
| :--- | ---: |
| plain F16 | 10.8198 +/- 0.1374 |
| rotated F16 | 10.8214 +/- 0.1374 |

+0.015% apart. That is the check that matters -- 91% of these weights had never
been rotated before.

**But the Q2_K benefit does not transfer.**

| OLMoE-1B-7B Q2_K | PPL |
| :--- | ---: |
| plain | 14.2224 +/- 0.1842 |
| rotated | 14.1211 +/- 0.1813 |

-0.71%: the right direction, but **inside the error bars** and roughly six times
smaller than the dense models' -4.3% and -4.7%.

## The dense numbers may not be significant either

This run used 200 chunks and recorded llama-perplexity's own error estimate. The
dense runs used **20 chunks and never recorded one**. At 200 chunks the error
here is about +/-0.18 on a PPL near 14; at 20 chunks it would be roughly
sqrt(10) larger, around +/-0.58, which is ~4% -- the same size as the -4.3%
effect that was being claimed.

So the one finding said to have survived two models may never have cleared its
own noise. Two models agreeing in sign is weak evidence: a coin lands the same
way twice 25% of the time. A re-run of Qwen3-4B Q2_K under this methodology is
the thing that settles it, and is in progress.

## Random signs made it worse

| Q2_K from (Qwen3-4B) | PPL |
| :--- | ---: |
| plain | 15.4422 |
| rotated, `sign_mode = identity` | **14.7720** |
| rotated + random signs, `sign_mode = explicit` | 15.6088 |

Opposite of the QuaRot/QuIP# argument that *randomness* is the active
ingredient. Here the plain Sylvester-Walsh matrix beat the randomised one, and
the randomised one was worse than no rotation at all. **Single seed, single
model -- given that the Q3/Q4 results turned out to be noise, treat this as
unreplicated.**

## Caveats

- Two models, both Qwen. No other family tested.
- **One corpus** (llama.cpp docs -- technical markdown). The effect could be
  corpus-specific; wikitext-2 would make these numbers comparable to published
  work and is the cheapest remaining check.
- One block size (512), one sign seed.
- **Only runs on the PrismML fork.** Upstream has no `prism.hadamard.*` support
  for weights -- it merged Hadamard for the KV cache
  ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038)), not for weights.

## What would make it a real finding

Ranked by information per unit cost:

1. **A different corpus** (wikitext-2) -- tests a live confound, costs a download.
2. **Several sign seeds** -- the random-sign result is the oddest thing here and
   rests on one seed.
3. **Block size** -- 1024 vs 512. Bigger blocks mix more values, so the
   outlier-spreading should strengthen if the mechanism is what we think.
4. **A non-Qwen model** -- tests architecture generality, costs ~16 GB.
