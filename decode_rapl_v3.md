# DECODE-RAPL: Migration Plan v2 $\rightarrow$ v3

## 1\. Goal

The v2 training report showed that our model is highly successful at predicting power (`R²=0.95`), but the validation curves indicated that the **reconstruction task is a distraction**. The model overfits on reconstruction, which conflicts with the power prediction task.

The goal of v3 is to **test this hypothesis** by creating a simpler, faster, and more focused model that optimizes *only* for power prediction.

-----

## 2\. Architectural Changes (v3)

We will remove the autoencoder's decoder and its associated loss, simplifying the model to a pure "Encoder-Predictor" architecture.

### What to Remove:

  * The **Decoder** network (`64 $\rightarrow$ 128 $\rightarrow$ 512 $\rightarrow$ 100`).
  * The `reconstruction_loss` component from the total loss function.

### What to Keep:

  * The **Encoder** (`100 $\rightarrow$ 512 $\rightarrow$ 128 $\rightarrow$ 64`).
  * The **Latent Space** (`64-dim`).
  * The **Power Head** (`64 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 1`).

### New v3 Architecture

The new architecture is a simple, feedforward network. The "latent space" is now just the "bottleneck" layer that connects the encoder and the predictor.

```
 Input (100)
    │
    ▼
 Encoder (100 → 512 → 128 → 64)
    │
    ▼
 Latent Vector "z" (64)
    │
    ▼
 Power Head (64 → 128 → 64 → 1)
    │
    ▼
 Power Prediction (1)
```

-----

## 3\. Training & Monitoring Changes

### Loss Function

  * The total loss is now *only* the power loss.
  * `Total Loss = Power Loss (MSE)`

### Monitoring

  * **This is the most critical change.** You must stop monitoring the combined loss.
  * **EarlyStopping:** The `monitor` parameter must be set to `validation_power_loss` (or whatever you call your validation MSE metric).
  * **ReduceLROnPlateau:** The `monitor` parameter must also be set to `validation_power_loss`.
  * **Logging:** Your training logs and plots should now *only* track `Training Power Loss` and `Validation Power Loss`. The messy reconstruction curve is gone.

-----

## 4\. Validation (Unchanged)

The simplification makes our validation tests *cleaner* and *more meaningful*.

1.  **Primary Metrics (R², MAE, MAPE):** These are still your main measure of success.
2.  **The "Power Knob" Test:** This test is now *better*.
      * The latent space `z` is no longer "conflicted" by the reconstruction task.
      * It is now a *pure* representation of the features needed for power prediction.
      * Analyzing `$\partial Power / \partial z$` to find the "Power Knobs" will be a more direct and meaningful analysis of your model's learned physics.
