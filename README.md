# ZClip: Adaptive Spike Mitigation for LLM Pre-Training


Official PyTorch Lightning implementation of our paper:

<b>ZClip: Adaptive Spike Mitigation for LLM Pre-Training</b>

[Abhay Kumar](https://www.linkedin.com/in/akanyaani/), [Louis Owen](https://www.linkedin.com/in/louisowen/), [Nilabhra Roy Chowdhury](https://www.linkedin.com/in/nilabhraroychowdhury/), [Fabian Güra](https://www.linkedin.com/in/guera/) 

BluOrion

[Paper](#)


---

## 🧠 Algorithm Overview

ZClip mitigates gradient spikes by maintaining running statistics of gradient norms using Exponential Moving Averages (EMA). At each training step, it updates both the mean and variance of the gradient norm without storing the full history. This allows the algorithm to adaptively respond to sudden changes in training dynamics.

When the current gradient norm significantly deviates from the recent trend, ZClip dynamically computes a clipping threshold based on the observed variance. This ensures that extremely large gradient updates—often responsible for loss spikes—are automatically suppressed without requiring static thresholds.

By continuously adjusting to the scale and variability of gradients throughout training, ZClip maintains both stability and learning efficiency even under high learning rates or aggressive schedules.

---

## 📉 Example Impact

<table>
<tr>
<td align="center">
<img src="./figures/3e3.png" width="400"/>
<br><b>Training Loss</b>
</td>
<td align="center">
<img src="./figures/lr_3e3_after.png" width="400"/>
<br><b>Gradient Norm after Clipping</b>
</td>
</tr>
</table>

---

## ⚙️ Implementation Details

Our code is built within the PyTorch Lightning framework, utilizing its callback system for seamless integration into the training pipeline. It is fully compatible with FSDP and requires no code changes to work out of the box.

You can also use ZClip directly with standard PyTorch by calling `.step(model)` after `loss.backward()` and before `optimizer.step()`.

---

## 🧪 Usage

### PyTorch
```python
from zclip import ZClip

zclip = ZClip(mode="zscore", alpha=0.97, z_thresh=2.5, clip_option="adaptive_scaling", max_grad_norm=1.0, clip_factor=1.0)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    zclip.step(model)
    optimizer.step()
```

### PyTorch Lightning
```python
from zclip_lightning_callback import ZClipLightningCallback

zclip_cb = ZClipLightningCallback(mode="zscore", alpha=0.97, z_thresh=2.5, clip_option="adaptive_scaling", max_grad_norm=1.0, clip_factor=1.0)

trainer = pl.Trainer(
    max_epochs=3,
    callbacks=[zclip_cb]
)

trainer.fit(model, dataloader)
```

---

## 🔍 ZClip Parameters

| Argument        | Description                                                                                                                                         | Default            |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `mode`          | Clipping mode. Options: <br> • `"zscore"` – Uses z‑score based clipping. <br> • `"percentile"` – Uses fixed threshold clipping defined as EMA mean plus (z_thresh × std). | `"zscore"`         |
| `z_thresh`      | Threshold value. In "zscore" mode, it sets the z‑score threshold; in "percentile" mode, it is used as the multiplier for std.                      | `2.5`              |
| `alpha`         | EMA smoothing factor for updating the gradient norm statistics.                                                                                    | `0.97`             |
| `clip_option`   | *(Only for "zscore" mode)* Clipping strategy: <br> • `"adaptive_scaling"` – Compute an adaptive threshold if the z‑score is high. <br> • `"mean"` – Clip to the EMA mean. | `"adaptive_scaling"` |
| `clip_factor`   | Constant Multiplier for the adaptive scaling threshold. A value between **0.3** and **0.7** yields more aggressive clipping, while a higher value (default `1.0`) is less aggressive. | `1.0`              |
| `max_grad_norm` | Optional maximum gradient norm to limit the clipping threshold.                                                                                     | `1.0`             |
| `warmup_steps`  | Number of steps to collect gradient norms for initializing the EMA statistics.                                                                     | `25`               |


---
## Aggressive Hyperparameter Settings

When training models with volatile gradients, noisy data, or when using curriculum learning strategies, more aggressive gradient clipping can be beneficial. In such scenarios, consider adjusting the following parameters:

- **`alpha`**:  
  The `alpha` parameter controls the smoothing of the EMA for gradient norm statistics. A lower value (e.g. around **0.90-0.95**) makes the EMA more responsive to recent gradients, which can be beneficial for rapidly changing gradient distributions. However, setting it too low might introduce noise into the EMA estimate, so it must be balanced carefully.

- **`z_thresh`**:  
  You may also consider reducing the `z_thresh` slightly (for example, from the default `2.5` to around **2.0**) to tighten the criteria for clipping further.

- **`clip_factor`**:  
  Lowering the `clip_factor` to a value between **0.3** and **0.7** will reduce the adaptive threshold in the `"adaptive_scaling"` mode, resulting in more aggressive clipping. This can help stabilize training by curbing large gradient spikes.

These settings are particularly useful in scenarios where the gradient distribution is highly dynamic. Adjust and monitor these hyperparameters based on your specific model, dataset, and training dynamics to achieve optimal performance.


---

## 📜 License
MIT License
