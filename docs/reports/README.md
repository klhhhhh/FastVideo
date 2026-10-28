# Experiment reports

Store reviewed, compact experiment reports and supporting CSV/JSON here. Keep
model weights, raw tensors, generated media and large profiler traces under
ignored `outputs/`; describe their location and retain source hashes in each
report bundle. Record failed checks and measurement limits alongside results.

| Report | Result |
| --- | --- |
| [H3 two-A6000 static placement, 2026-10-28](h3/2026-10-28-a6000/REPORT.md) | No observed placement speedup; fixed-mask oracle passed; strict model-output reproducibility failed. |
