# TriSplat++ model artifacts

The reference LGTM appearance model is:

```text
/root/data/haoxuan/TriSplat/outputs/exp_lgtm10k_train/2026-08-28_11-58-26/checkpoints/render_step_002700.ckpt
```

It is a 149 MiB Lightning checkpoint (global step 2700) containing the DA3
TSDPT-compatible geometry output head and the trainable LGTM texture head.
The SHA-256 digest of the source artifact is:

```text
9fa4c3011bf8aff893a18768330855a4c27b19c413c3ff11cba40ad653331981
```

The binary is tracked with Git LFS because GitHub's normal object limit is
100 MiB. If the remote does not provide LFS, keep the artifact outside Git and
export its path instead:

```bash
export TRISPLATPP_CHECKPOINT=/path/to/render_step_002700.ckpt
```

The checked-in LFS filename is `weights/trisplatpp_lgtm_step2700.ckpt`.
