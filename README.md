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

## Leading hypothesis: identity signs are the problem

We used `sign_mode = identity` — a bare Sylvester-Walsh matrix, which is fixed
and highly structured. Prism uses `sign_mode = explicit` with **28,672 random
+/-1 values** (`sign_widths = [5120, 6144, 17408]`).

Those signs are the point. In this family of methods (QuaRot, QuIP#) it is a
*random* orthogonal rotation that Gaussianises the weight distribution via the
central limit theorem, which is what makes a three-value grid viable. `diag(s).H`
is random; `H` alone is not.

The runtime already supports it — `llama-graph.cpp` applies
`ggml_mul(ctx0, cur_mm, t.signs)` before the Hadamard.

**Next step:** generate random sign vectors, fold `diag(s)` into the weights
alongside `H`, and declare `sign_mode = explicit`. Same fast test loop.

Beyond that, Prism almost certainly also calibrates (cf. PT^2-LLM's iterative
ternary fitting), so "coherent but worse than theirs" remains the realistic
ceiling for this approach.

## Files

| file | what |
| :--- | :--- |
| `rotate.py` | the rotation tool |

Run with the venv at `~/bonsai/rotvenv` and
`PYTHONPATH=~/bonsai/llama.cpp/gguf-py`.
