# TriSplat++ model artifacts

The reference LGTM appearance model is:

```text
/root/data/haoxuan/TriSplat/outputs/exp_lgtm10k_train/2026-08-28_11-58-26/checkpoints/render_step_002700.ckpt
```

It is a 149 MiB Lightning checkpoint (global step 2700) containing **only the
trained DA3/TSDPT output head**. Its `state_dict` has 92 tensors under
`encoder.da3.model.gs_head.*`; it does not contain the DA3-GIANT backbone,
camera encoder, depth/feature branches, or a complete standalone model. The
LGTM texture-remapping implementation lives in the repository and must be
constructed by the TriSplat++ code path.

This is therefore a head-only artifact and **must be paired with the complete
DA3-GIANT-1.1 checkpoint** at runtime. Point `DA3_CHECKPOINT` (or the
renderer’s `--da3-checkpoint`) at the full DA3 model directory, for example:

```bash
export DA3_CHECKPOINT=/path/to/Depth-Anything-3/checkpoints/DA3-GIANT-1.1
export TRISPLATPP_CHECKPOINT=$PWD/weights/trisplatpp_lgtm_step2700.ckpt
```

The SHA-256 digest of this head-only artifact is:

```text
9fa4c3011bf8aff893a18768330855a4c27b19c413c3ff11cba40ad653331981
```

The binary is tracked with Git LFS because GitHub's normal object limit is
100 MiB. If the remote does not provide LFS, keep the artifact outside Git and
export its path instead:

```bash
export TRISPLATPP_CHECKPOINT=/path/to/render_step_002700.ckpt
```

The checked-in LFS filename is `weights/trisplatpp_lgtm_step2700.ckpt`. The
same file is published in the Hugging Face model repository
[`hhx112340/trisplatplus`](https://huggingface.co/hhx112340/trisplatplus).
