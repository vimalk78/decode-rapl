# DECODE-RAPL: v4 Architecture (1D-CNN Encoder + Power Head)

This architecture replaces the initial MLP layers of the v3 encoder with 1D Convolutional layers to better process the temporal information within the delay-embedded input.

-----

## 1\. Input Reshaping

  * **Input:** Original flattened vector `(batch, 100)`.
  * **Action:** Reshape the input to `(batch, num_features, sequence_length)`.
  * **Shape:** **`(batch, 4, 25)`**
      * `num_features = 4` (user%, sys%, iowait%, log\_ctx)
      * `sequence_length = 25` (delay embedding dimension `d`)

-----

## 2\. 1D Convolutional Block

Apply `Conv1d` layers to act as sliding filters across the 25 time steps for each feature channel.

  * **Layer 1:**
      * `nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)`
      * `nn.ReLU()`
      * *Output Shape:* `(batch, 32, 25)`
  * **Layer 2:**
      * `nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)`
      * `nn.ReLU()`
      * *Output Shape:* `(batch, 64, 25)`
  * **Optional Pooling:**
      * `nn.MaxPool1d(kernel_size=2)`
      * *Output Shape:* `(batch, 64, 12)` (Reduces sequence length)

-----

## 3\. Flatten

  * **Action:** Flatten the output tensor from the convolutional block.
  * **Shape:** `(batch, 64 * 12)` = **`(batch, 768)`** (Assuming pooling was used)

-----

## 4\. MLP Layers (Post-Convolution)

Process the extracted temporal features using fully connected layers.

  * **Layer 1:** `nn.Linear(768, 128)` -\> `nn.ReLU()`
  * **Layer 2:** `nn.Linear(128, 64)` -\> **Latent Space**
      * *Output Shape:* `(batch, 64)`

-----

## 5\. Power Head (Unchanged)

Connect the standard Power Head MLP to the 64-dimensional latent space output.

  * `nn.Linear(64, 128)` -\> `nn.ReLU()`
  * `nn.Linear(128, 64)` -\> `nn.ReLU()`
  * `nn.Linear(64, 1)` -\> **Power Prediction**
      * *Output Shape:* `(batch, 1)`

-----

## 6\. Diagram

```
Input (batch, 100) -> Reshape -> (batch, 4, 25)
    │
    ▼
Conv1d(in=4, out=32, kernel=3, padding=1) -> ReLU
    │ Shape: (batch, 32, 25)
    ▼
Conv1d(in=32, out=64, kernel=3, padding=1) -> ReLU
    │ Shape: (batch, 64, 25)
    ▼
MaxPool1d(kernel=2)
    │ Shape: (batch, 64, 12)
    ▼
Flatten
    │ Shape: (batch, 768)
    ▼
Linear(in=768, out=128) -> ReLU
    │
    ▼
Linear(in=128, out=64) ---------> Latent Space (batch, 64)
                                     │
                                     ▼
                                  Power Head:
                                  Linear(in=64, out=128) -> ReLU
                                     │
                                     ▼
                                  Linear(in=128, out=64) -> ReLU
                                     │
                                     ▼
                                  Linear(in=64, out=1)
                                     │
                                     ▼
                                  Power Output (batch, 1)
```


-----

### 6\.1 v4.1 




```
+--------------------------+
| Input                    |
| (batch, 100)             |
+--------------------------+
            |
            v
+--------------------------+
| Reshape                  |
| (batch, 4, 25)           |
+--------------------------+
            |
            v
+--------------------------+
| Layer: Conv1d(4->32, k=5)|
| Layer: BatchNorm1d(32)   |
| Layer: ReLU              |
| Shape: (batch, 32, 25)   |
+--------------------------+
            |
            v
+--------------------------+
| Layer: Conv1d(32->64, k=5)|
| Layer: BatchNorm1d(64)   |
| Layer: ReLU              |
| Shape: (batch, 64, 25)   |
+--------------------------+
            |
            v
+--------------------------+
| Layer: MaxPool1d(k=2)    |
| Shape: (batch, 64, 12)   |
+--------------------------+
            |
            v
+--------------------------+
| Flatten                  |
| Shape: (batch, 768)      |
+--------------------------+
            |
            v
+--------------------------+
| Layer: Linear(768->128)  |
| Layer: ReLU              |
| Layer: Dropout(0.3)      |
+--------------------------+
            |
            v
+--------------------------+
| Layer: Linear(128->64)   | --> Latent Space (batch, 64)
+--------------------------+
            |
            v
+--------------------------+
| Layer: Linear(64->128)   |
| Layer: ReLU              |
| Layer: Dropout(0.3)      |
+--------------------------+
            |
            v
+--------------------------+
| Layer: Linear(128->64)   |
| Layer: ReLU              |
+--------------------------+
            |
            v
+--------------------------+
| Layer: Linear(64->1)     |
+--------------------------+
            |
            v
+--------------------------+
| Power Output             |
| (batch, 1)               |
+--------------------------+
```

## 7\. Implementation Notes

  * Ensure the input tensor is correctly reshaped to `(batch, 4, 25)` within the model's `forward` method.
  * The number of convolutional filters (32, 64), kernel sizes, padding, and pooling choices are hyperparameters that can be tuned. This configuration provides a solid starting point.
  * Monitor training closely as this architecture has more parameters than the pure MLP version. Regularization (Dropout, Weight Decay) might be important.
