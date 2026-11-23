<h1 align="center">Self-Supervised Representation Learning for DNA Sequence Classification</h1>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-SWITCH-4E9A06?style=flat-square" alt="Switch to Chinese"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/English-CURRENT-1F4E79?style=flat-square" alt="English">
</p>

<p align="center">
  Masked-reconstruction pretraining of a CNN encoder on unlabelled genomic fragments, then linear probing on a four-species classification task.
</p>

## Why this exists

Supervised learning on genomic sequences is bottlenecked by labels, not by data. This project tests the other order: pretrain an encoder on the sequences themselves, freeze it, and ask how much of the task a single linear layer can recover. If the answer is close to the supervised baseline, the representation carries the signal and labels are only needed at the end.

The repository includes the training logs, the projection plots, and a written post-mortem of the period when the classification head produced nothing but chance-level output.

## Results

| Stage | Metric | Value |
|---|---|---|
| Pretraining, 10 epochs | train loss | 4.0609 to 0.6103 |
| | validation loss | 3.8067 to 0.5339 |
| Downstream, 50 epochs | **validation accuracy** | **0.6825** |
| | precision / recall / F1 | 0.6822 / 0.6871 / 0.6830 |
| | chance baseline, 4 classes | 0.2500 |

The pretraining loss falls cleanly, which is the trap in this setup: a masked-reconstruction objective can converge while producing a representation that is useless downstream. That gap is what the post-mortem is about.

## Method

1. **Tokenise** genomic fragments into fixed-length windows.
2. **Pretrain** a CNN encoder with a masked-reconstruction objective: randomly mask a proportion of positions, reconstruct them from context.
3. **Freeze** the encoder and discard the reconstruction head.
4. **Probe** with a single linear layer trained on the labelled subset.
5. **Inspect** the representation with PCA and t-SNE, and compare against the chance baseline.

Freezing is deliberate. Fine-tuning would conflate the quality of the representation with the capacity of the downstream model; a linear probe on frozen features gives a cleaner read.

## Project structure

```
pretraining/          encoder and masked-reconstruction training
classification/       frozen-feature linear probe
visualizations/       PCA and t-SNE projections
scripts/              rerun_classification.py, utilities
experiment_results.json
```

## Quick start

```bash
pip install -r requirements.txt

# Full pipeline: pretrain, probe, visualise (slow on CPU)
python scripts/train.py

# With a pretrained checkpoint, skip pretraining
python scripts/rerun_classification.py

# Tests
pytest
```

## Output files

| File | Contents |
|---|---|
| `experiment_results.json` | All metrics, per-epoch losses and validation accuracy, config snapshot |
| `pretraining/final_model.pt` | Pretrained weights, reusable by `scripts/rerun_classification.py` |
| `visualizations/dna_features_pca.png` | PCA projection of the learned representation |
| `visualizations/dna_features_tsne.png` | t-SNE projection |

## Post-mortem: why the classification head sat at chance

### Symptom

Validation accuracy stayed at 0.25 across epochs, exactly the four-class chance rate, while training loss on the probe continued to decrease. Loss curves looked healthy; the model was learning nothing transferable.

### Localisation

Three hypotheses were tested in isolation rather than by tuning everything at once:

1. **Representation collapse.** If the encoder mapped all inputs to a near-constant vector, a linear probe could only ever output the prior. Checked by inspecting the variance of frozen features and the PCA projection.
2. **Masking ratio too aggressive.** If too many positions were masked, reconstruction became unsolvable from context and the encoder learned noise. Checked by sweeping the ratio and watching reconstruction loss against probe accuracy.
3. **Warmup schedule.** If the learning rate spent its warmup before the encoder had moved meaningfully, the representation would be fixed at initialisation. Checked by removing warmup and by starting the probe from a mid-training checkpoint.

The order matters: each hypothesis has a distinct signature, and testing them one at a time is what makes the result attributable. The exact finding and the change that resolved it are recorded in the repository history and the write-up alongside the code.

## Known limitations

- Four species only; the transfer claim does not extend to a larger taxonomy without further work.
- The probe is linear by design. A stronger downstream head would raise the number and tell you less about the representation.
- Pretraining was capped at 10 epochs for CPU feasibility. The loss had not plateaued, so the reported accuracy is a floor, not a ceiling.
- Single seed. The 0.68 figure should be read with the variance of repeated runs in mind, which this repository does not provide.

## Licence

MIT. See [LICENSE](LICENSE).
