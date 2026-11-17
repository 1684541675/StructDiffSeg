# SDF residual fusion experiment

This is a new experiment, not a replacement for baseline/SDF/legacy GCF.
Existing source files and experiment results are unchanged.

## What changes

At encoder stage 2, first execute the existing concatenation + 1x1 convolution:

    F_old = Conv1x1(concat(F_diff, F_geometry))
    H = GELU(InstanceNorm(Conv3x3(concat(F_geometry, resize(predicted_SDF)))))
    gate = sigmoid(Conv1x1(concat(F_old, H)) + Linear(time_embedding))
    F_new = F_old + gamma * gate * H

All other scales retain their original fusion, including bottleneck affinity
fusion. The SDF head, SDF loss, DDIM process and final addition of auxiliary
segmentation logits are unchanged. This isolates the fusion change; it does
not add a new distance loss or guarantee better scores.

Gamma starts at zero. With the same shared seed, the initial predictions equal
the SDF baseline exactly. Gamma receives a gradient on the first update; the
other residual parameters begin receiving gradients after gamma leaves zero.
The supervised SDF head trains from the first update. Gamma is signed and may
become negative. It is not a probability or a measure of model quality.

## Run

Run these PowerShell commands from the repository root. The existing working
environment is `D:\unet_env\Scripts\python.exe`. The default `D:\ai\python.exe`
failed to load its user-site PyTorch DLL during verification.

Start with the same 64x64 settings as the existing experiments to compare the
fusion change. This command starts a new 100-epoch run; it has not been run by
the implementation task:

```powershell
& D:\unet_env\Scripts\python.exe train_gcf_residual_2d.py --dataset glas --data-path D:/code/Python/MedSeg/glas --variant gcf_residual --mode train --size 64 --epochs 100 --batch-size 16 --seed 42 --output-dir runs/glas_gcf_residual_64 --device cuda
```

After training, evaluate the validation-selected checkpoint with the NEW test
entry point (the legacy test entry point cannot load this architecture):

```powershell
& D:\unet_env\Scripts\python.exe test_gcf_residual_2d.py --checkpoint runs/glas_gcf_residual_64/best.pt --device cuda
```

Use a distinct, empty output directory for each experiment. Existing nonempty
directories are rejected. Resume a new-version run with the same arguments plus
`--resume runs/glas_gcf_residual_64/last.pt`; do not resume legacy checkpoints.

Optional variants, each with a separate output directory:

| Variant | SDF input to residual | Time input to residual |
|---|---|---|
| `gcf_residual` | Predicted SDF | Time embedding |
| `gcf_residual_no_sdf` | Zeros | Time embedding |
| `gcf_residual_no_time` | Predicted SDF | Zeros |

The ablations keep the same parameter shapes and shared initialization. SDF
supervision remains enabled in ALL variants. The no-time variant removes time
only from this residual block; the diffusion backbone still uses time.

Training logs contain mean batch losses (`loss_diffusion`, `loss_geometry`,
`loss_sdf`), gradient norms from the existing SDF training step, and the learned
`residual_gamma`. Metrics and test PNGs retain the previous evaluation protocol.

## Verification

```powershell
python -m unittest discover -s tests -p test_gcf_residual.py -v
```

Checks cover initial DDIM equivalence to the SDF baseline, gradient flow from
diffusion predictions into the SDF head after the residual opens, MC repetition,
conditioning ablations, synthetic training, checkpoint reload/resume, and held-out
test export. Synthetic checks establish that the implementation runs, not that
real-data segmentation performance improves.
