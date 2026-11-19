# BC-NO: Boundary-Capacitance Neural Operators

Code accompanying the manuscript

> T. Liu, L. Liu, C. Zhu, *Boundary-Capacitance Neural Operators for High-Contrast Multiscale Homogenization*, prepared for *Computer Methods in Applied Mechanics and Engineering* (CMAME).

BC-NO learns RVE maps for high-contrast thermal and elastic homogenization. Constant-coefficient Green layers, normalized residuals, and connected-component capacitance pooling feed a dipole / virtual-work readout that keeps constitutive averaging contrast-stable.

Large `.pt` packs are **not** shipped: Colab runners regenerate fiber / bridge / laminate / FE² data on the fly. Compact CT morphology specs (`.npz`) are included under `data/`.

## Layout

```
src/                 models, RVE operators, baselines, paper-suite runners
colab/               Colab runbooks (V7/V8); colab/flat/ is a one-folder upload set
data/                compact CT / Ruhr morphology specs + empirical mechanism JSON
results/             frozen JSON tables from completed suite versions (v1–v6)
paper/               CMAME manuscript sources + figures
examples/            import smoke check
```

## Quick start (local import check)

```bash
pip install -r requirements.txt
python examples/smoke_import.py
```

## Colab (recommended for training)

1. Upload everything in `colab/flat/` (and optionally `data/rve_3d_ruhr64_compact.npz`) into `/content`.
2. Follow `colab/COLAB_V8_RUN.txt` for the four scale / engineering parts (A–D), or older `COLAB_V7_RUN.txt` for matched-accuracy / energy / curves add-ons.

```bash
%cd /content
!python /content/run_bcno_v8_scale_engineering.py --root /content --only A
```

Core paper suite (Exp.1–14 style):

```bash
!python /content/run_bcno_paper_suite.py
```

## Status

v8 scale-engineering runs (Exp.11 3-seed, enlarged FE², 64³ 3-seed, expanded Exp.1/4b) may still be in progress; results JSON will be added under `results/` when available.

## License

MIT — see `LICENSE`.
