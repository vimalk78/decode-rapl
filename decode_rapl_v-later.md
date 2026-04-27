Here is a design document for the v4 model, which incorporates the adversarial head to achieve your final goal of machine generalization.

-----

# DECODE-RAPL: v4 Design (Multi-Machine Generalization)

## 1\. Goal

The v2 and v3 models are designed to be high-accuracy power predictors for a *single machine*. They learn the specific power characteristics of the CPU they were trained on.

The final goal, "to decode RAPL," requires a model that learns the *universal power characteristics of an architecture* (e.g., "Intel 10th Gen Core"). This model must be able to predict power on a new, unseen CPU of that same architecture.

To do this, the model's latent space must become **machine-invariant**. It must learn to represent *only* the physics of the workload (proxying frequency, etc.) while actively **ignoring** machine-specific "noise" (like differences in idle power, specific voltage-per-core, etc.).

The v4 model achieves this by adding an **Adversarial Head** to the successful v3 (Encoder-Predictor) architecture.

-----

## 2\. v4 Architecture

The v3 model (`Encoder $\rightarrow$ Power Head`) is our new baseline. The v4 model adds a second "head" that branches off the same latent space. This new head's job is to *fight* the Encoder.

```
 Input (100)
    │
    ▼
 Encoder (100 → 512 → 128 → 64)
    │
    ▼
 Latent Vector "z" (64)
    │
    ├─► Power Head (64 → 128 → 64 → 1)
    │      │
    │      ▼
    │   Power Prediction (for power_loss)
    │
    └─► Adversarial Head (Classifier) (64 → 128 → num_machines)
           │
           ▼
        Machine Prediction (for adversarial_loss)
```

### New Components

  * **Adversarial Head (Discriminator):** A simple MLP classifier (e.g., `64 $\rightarrow$ 128 $\rightarrow$ num_machines`).
  * **`num_machines`:** The number of unique CPUs in your training set (e.g., `3` if you have data from 3 machines).
  * **Gradient Reversal Layer (GRL):** This is a special "pseudo-layer" placed between the Encoder and the Adversarial Head. It passes the data forward normally, but when backpropagating, it *reverses the sign of the gradient*.

-----

## 3\. Training & Loss Function

This model is trained by two competing objectives:

1.  **The *Primary* Goal (Power Prediction):**

      * **Path:** `Encoder $\rightarrow$ Power Head`
      * **Loss:** `power_loss = MSE(predicted_power, actual_power)`
      * **Goal:** The Encoder and Power Head work *together* to **minimize** this loss. This forces the latent space `z` to *contain* all information needed to predict power.

2.  **The *Adversarial* Goal (Machine Anonymization):**

      * **Path:** `Encoder $\rightarrow$ GRL $\rightarrow$ Adversarial Head`
      * **Loss:** `adversarial_loss = CrossEntropy(predicted_machine, actual_machine_id)`
      * **Goal (Two-Part):**
          * The **Adversarial Head** tries to **minimize** this loss (i.e., it gets good at guessing the machine ID from `z`).
          * The **Encoder** (because of the GRL) tries to **maximize** this loss (i.e., it changes `z` to be as confusing as possible for the Adversarial Head).

The Encoder is forced to find a solution that **minimizes `power_loss`** while **maximizing `adversarial_loss`**. The only way to do this is to create a latent space `z` that is **excellent for predicting power** but **useless for identifying the machine**.

-----

## 4\. Data Requirements

**This is the most critical change.** The v4 model *cannot* be trained on your current dataset.

  * **New Data:** You must run your `run_workloads.sh` data generation script on **multiple (e.g., 3-5) different bare-metal machines** that share the target architecture.
  * **New Labels:** When pre-processing, every sample vector must be given **two** labels:
    1.  `power_label` (a float, e.g., `65.4`)
    2.  `machine_id` (an integer, e.g., `0`, `1`, `2`, ...)
  * **Data Split:** The "Global Shuffle" method is still correct. You will shuffle all (e.g., 15+ million) samples from all machines into one giant dataset before splitting into train/val/test.

-----

## 5\. Validation & "Decoding"

Your final R²/MAE/MAPE metrics are still the primary measure of success. But to prove *generalization*, your validation changes:

1.  **Monitor Discriminator Accuracy:** The key new metric is the validation accuracy of the Adversarial Head. If you have 3 machines, its accuracy should drop from 100% and hover around **33% (random chance)**. This proves your Encoder is successfully "anonymizing" the latent space.

2.  **t-SNE / UMAP Plot (The "Aha\!" Moment):**

      * Plot your validation latent space `z` in 2D.
      * **Color by Power:** You should still see a smooth gradient (proving power prediction works).
      * **Color by `machine_id`:** You should see **no distinct clusters**. The colors (e.g., red, green, blue for 3 machines) should be completely mixed together. This is the visual proof of a machine-invariant latent space.

3.  **The "Power Knob" Test (The Final Proof):**

      * This test remains your ultimate validation.
      * You must confirm that the *same* "Power Knob" (e.g., `z_12`) from your trained Encoder correlates with the *native* `cpu_frequency` on **all** your test machines (Machine A, Machine B, and Machine C).

4.  **The Final Exam:**

      * Train the v4 model on data from Machines A, B, and C.
      * Test its R²/MAE on data from a **new, unseen Machine D**.
      * If the R²/MAE remains high, you have successfully "decoded" the RAPL for that architecture.
