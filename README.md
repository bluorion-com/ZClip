# ZClip: Adaptive Gradient Clipping

![ZClip Overview](ZClip/data/zclip.png)

ZClip is an adaptive gradient clipping strategy designed to reduce loss spikes during large language model (LLM) pretraining. It combines exponential moving average (EMA) tracking with statistical anomaly detection to determine clipping thresholds dynamically.

[📄 Read the full paper](https://your-paper-link-here.com)

---

## 🧠 Algorithm Overview

ZClip mitigates gradient spikes by tracking the gradient norm across steps using Exponential Moving Average (EMA). It supports two detection modes:

- **Z-Score Mode**: Detects gradient outliers based on how many standard deviations they are from the mean.
- **Percentile Mode**: Clips any gradient norm that exceeds the N-th percentile of the expected distribution.

This adaptive behavior allows ZClip to be more robust than fixed-threshold clipping, especially during dynamic training phases or with high learning rates.

---

## ⚙️ Implementation Details

Our code is built within the PyTorch Lightning framework, utilizing its callback system for efficient integration into the training pipeline.

You can also use ZClip directly with standard PyTorch by calling `.step(model)` after `loss.backward()` and before `optimizer.step()`.

---

## 🧪 Usage

### PyTorch
```python
from zclip import ZClip

zclip = ZClip(mode="zscore", alpha=0.97, z_thresh=2.5)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    zclip.step(model)
    optimizer.step()
```

### PyTorch Lightning
```python
from zclip_callback import ZClipCallback

zclip_cb = ZClipCallback(mode="zscore", alpha=0.97, z_thresh=2.5)

trainer = pl.Trainer(
    max_epochs=3,
    callbacks=[zclip_cb]
)

trainer.fit(model, dataloader)
```

---

## 🔍 ZClip Parameters

| Argument        | Description                                                 | Default |
|----------------|-------------------------------------------------------------|---------|
| `mode`         | "zscore" or "percentile" clipping mode                      | zscore  |
| `z_thresh`     | Z-score threshold (used if mode=zscore)                    | 2.5     |
| `percentile`   | Percentile value (used if mode=percentile)                 | 0.99    |
| `alpha`        | EMA smoothing factor                                        | 0.97    |
| `clip_factor`  | Multiplier for std when clipping                           | 1.0     |
| `warmup_steps` | Number of steps to initialize EMA statistics               | 25      |

---

## 📊 Benefits

- Prevents catastrophic loss spikes
- Enables higher learning rates
- No manual tuning of static thresholds
- Compatible with PyTorch and PyTorch Lightning

---

## 📜 License
MIT License
