# Waterbirds Phase 0: one-model ERM baseline

This project trains exactly one ImageNet-pretrained torchvision ResNet-50 with ordinary cross-entropy. It uses the official Waterbirds metadata split and does not use group information during training. Group labels are used only for reporting.

## Prepare data

```powershell
python scripts/download_waterbirds.py --data-dir F:/Waterbirds-repair-selection/data/waterbirds
```

The official `metadata.csv` encodes `y` (`0=landbird`, `1=waterbird`), `place` (`0=land`, `1=water`), and `split` (`0=train`, `1=val`, `2=test`). The evaluation group is `2*y + place`.

## Train and evaluate

```powershell
python train_baseline.py --config configs/waterbirds_baseline.yaml
```

Outputs are written to `F:/Waterbirds-repair-selection/outputs/baseline_seed0`: `config.json`, `dataset_statistics.json`, `best_model.pt`, `metrics.json`, and `train.log`.

The best epoch is selected only on validation overall accuracy; group labels never affect training or checkpoint selection. Test metrics are computed once after training using that checkpoint. No official split is changed.

Pretrained torchvision weights are cached under F:/Waterbirds-repair-selection/data/torch_cache, not on C:.

