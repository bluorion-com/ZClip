# ZClip: Adaptive Gradient Clipping

ZClip is a PyTorch and PyTorch Lightning-compatible module for adaptive gradient clipping, using exponential moving averages and z-score/percentile-based anomaly detection. It stabilizes large-scale training by preventing gradient explosions and loss spikes.

---

## 🔧 Features
- EMA-based gradient norm tracking
- Z-score and percentile-based clipping modes
- Lightning callback support for easy integration
- Lightweight, no model modification required

---

## 📦 Installation
```bash
pip install torch scipy pytorch-lightning
```
Clone this repo and use `zclip.py` and `zclip_callback.py` in your project.

---

## 🧠 Usage

### PyTorch (Vanilla)
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

## ⚙️ ZClip Arguments
| Argument        | Description                                                 | Default |
|----------------|-------------------------------------------------------------|---------|
| `mode`         | "zscore" or "percentile" clipping mode                      | zscore  |
| `z_thresh`     | Z-score threshold (used if mode=zscore)                    | 2.5     |
| `percentile`   | Percentile value (used if mode=percentile)                 | 0.99    |
| `alpha`        | EMA smoothing factor                                        | 0.97    |
| `clip_factor`  | Multiplier for std when clipping                           | 1.0     |
| `warmup_steps` | Number of steps to initialize EMA statistics               | 25      |

---

## 📈 Example Output
ZClip helps reduce catastrophic loss spikes and improves convergence speed by adaptively managing gradient norms during training.

---

## 📜 License
MIT License