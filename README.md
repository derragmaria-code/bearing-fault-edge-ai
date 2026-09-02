# Bearing Fault Diagnosis — Edge-Deployable 1D-CNN

Vibration-signal fault classification for rolling-element bearings, built for
low-power edge deployment: a compact 1D-CNN trained on CWRU vibration data,
compressed via structured pruning + INT8 post-training quantization, then
stress-tested for cross-dataset generalization on the Paderborn (PU) bearing
dataset.

## Pipeline

The notebook (`notebooks/bearing_fault_diagnosis.ipynb`) runs end to end:

1. **Signal extraction & windowing** — pulls the `*_DE_time` channel from CWRU
   `.mat` files, slices into fixed-length windows, labels by fault type.
2. **Train/test split & normalization** — stratified group split (by source
   file) to avoid leakage between windows of the same recording.
3. **Baseline 1D-CNN** — small-on-purpose reference model.
4. **TFLite conversion & evaluation** — FP16 and INT8 post-training
   quantization, each re-evaluated against the FP32 baseline.
5. **Structured filter pruning** — coarse then fine-grained search over
   filter-keep ratios, with fine-tuning after each prune.
6. **Latency & footprint benchmarking** — per-sample inference latency and
   memory footprint across all variants, collated into a final comparison
   table (size / accuracy / macro-F1).
7. **Deployment export** — writes the chosen INT8 `.tflite` model plus
   `model_metadata.json` (class labels, window size, input/output
   quantization scale & zero-point) to `deployment/`.
8. **Best-candidate quantization** — INT8 quantization of the selected pruned
   model (`Pruned_50pct`).
9. **Taylor-expansion pruning ablation** — compares first-order Taylor filter
   importance (Molchanov et al.) against the L2-norm baseline used earlier.
10. **Cross-dataset generalization** — same pipeline applied to the Paderborn
    PU dataset to check how well the CWRU-trained model transfers.
11. **Domain-shift fixes** — global (dataset-level) normalization, Hilbert
    envelope demodulation, and MMD-based domain adaptation, with a
    lambda=0 control run to isolate what's actually helping.
12. **Interactive demo** — inline `ipywidgets` UI to run inference on either
    an uploaded CWRU `.mat` file or a synthetic illustrative signal.

## Repo structure

```
bearing-fault-diagnosis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── bearing_fault_diagnosis.ipynb   # full pipeline, steps 1–12
├── demo/
│   └── ipywidgets_demo.py              # standalone copy of the notebook's demo cell
└── deployment/                         # populated by Step 7 (not committed — see .gitignore)
    ├── <variant>.tflite
    └── model_metadata.json
```

## Requirements

```bash
pip install -r requirements.txt
```

Datasets (not included — download separately):
- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- [Paderborn University bearing dataset](https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter)

## Running the demo

The demo cell (Step 12) reuses the model already loaded in memory from Step 7
(`DEPLOY_MODEL_BYTES`, `metadata`) — no file I/O needed if you're running the
full notebook top to bottom. To run it standalone in another notebook, load
those two objects yourself (from `deployment/`) and paste in
`demo/ipywidgets_demo.py`.

## Notes

- The synthetic-signal mode in the demo generates illustrative
  fault-patterned signals (periodic impulses at approximate fault
  frequencies) for exploring the model without a real `.mat` file on hand —
  it is not real bearing data and shouldn't be used to judge real-world
  accuracy.
- Some notebook comments are in French.

