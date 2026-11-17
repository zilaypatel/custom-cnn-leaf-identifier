# Indian Herbal Leaf Classification Using a Custom CNN

An end-to-end Deep Learning Computer Vision project for the automated identification and medicinal classification of Indian herbal plant leaves.

Based on the research paper:  
**"Leaf Classification Using A Custom CNN"**  
*Kishan Patel, Jayesh Shinde, Meghraj Patel, and Jash Dhimmar*  
*Asha M. Tarsadia Institute of Computer Science and Technology*

---

## 1. Project Abstract

Indian herbal plants have served as the foundation of traditional Ayurvedic medicine for centuries. However, manual botanical classification of plant species requires specialized expertise and is error-prone due to variations in lighting, leaf morphology, and environmental damage.

This project implements an automated AI vision system using a **Custom Convolutional Neural Network (CNN)** capable of classifying five prominent Indian herbal plant leaf species from digital images:
1. **Lemon** (*Citrus limon*)
2. **Tulsi** (*Ocimum tenuiflorum*)
3. **Mango** (*Mangifera indica*)
4. **Money Plant** (*Epipremnum aureum*)
5. **Rosa-senensis / Hibiscus** (*Hibiscus rosa-sinensis*)

The deep learning model achieves high classification accuracy (**91.1%** test accuracy reported in the research benchmark) by extracting hierarchical spatial features directly from raw image pixels without manual hand-crafted feature engineering.

---

## 2. Botanical Dataset & Preprocessing

### 2.1 Botanical Classes & Taxonomic Details

| Class Directory | Common Name | Botanical / Scientific Name | Family | Key Medicinal Uses |
| :--- | :--- | :--- | :--- | :--- |
| `Lemon/` | Lemon / Nimbu | *Citrus limon* | Rutaceae | High Vitamin C; antioxidant, digestive aid, antimicrobial |
| `Tulsi/` | Holy Basil / Tulsi | *Ocimum tenuiflorum* | Lamiaceae | Adaptogen; respiratory remedy, immunity booster, antiviral |
| `Mango/` | Mango / Aam | *Mangifera indica* | Anacardiaceae | Rich in mangiferin & polyphenols; traditional blood sugar control |
| `money/` | Money Plant | *Epipremnum aureum* | Araceae | NASA clean air study recognized air purifier; VOC absorber |
| `Rosa-Senensis/` | China Rose / Gurhal | *Hibiscus rosa-sinensis* | Malvaceae | Hair growth promoter, cooling agent, anti-hypertensive |

### 2.2 Image Preprocessing & Augmentation Pipeline

1. **Spatial Resizing**: All input images are standardized to **128 × 128 × 3** (RGB).
2. **Min-Max Rescaling**: Pixel values are normalized from $[0, 255]$ to the interval $[0.0, 1.0]$:

   $$
   x_{\text{norm}} = \frac{x}{255.0}
   $$

3. **Data Augmentation**: To improve model generalization and prevent overfitting on minority classes, the training pipeline applies:
   - Random Horizontal Flip
   - Random Rotation ($\pm 15\%$)
   - Random Brightness adjustment ($\pm 15\%$)
   - Random Zoom ($\pm 10\%$)
4. **Class Weighting & Balancing**: The training pipeline calculates inverse class frequency weights:

   $$
   w_c = \frac{N_{\text{total}}}{K \cdot N_c}
   $$

   Where $N_{\text{total}}$ is total images, $K$ is the number of classes (5), and $N_c$ is the image count for class $c$. This ensures minority classes (`money`, `Mango`, `Tulsi`) exert proportional gradient influence.

---

## 3. Custom CNN Architecture & Mathematical Formulation

The neural network utilizes a sequential feedforward convolutional architecture optimized for fine-grained leaf boundary and venation pattern extraction.

```
Input Image (128x128x3)
        │
┌───────▼──────────────────────────┐
│  Conv2D (32 filters, 3x3) + ReLU │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  MaxPooling2D (2x2)              │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  Conv2D (64 filters, 3x3) + ReLU │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  MaxPooling2D (2x2)              │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  Conv2D (128 filters, 3x3)+ ReLU │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  MaxPooling2D (2x2)              │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  Flatten Layer                   │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  Dense (256 units) + ReLU        │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  Dropout (rate = 0.5)            │
└───────┬──────────────────────────┘
┌───────▼──────────────────────────┐
│  Dense (5 units) + Softmax       │
└──────────────────────────────────┘
```

### 3.1 Layer Configuration Table

| Layer Type | Filter / Units | Kernel / Pool Size | Activation | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **InputLayer** | — | — | — | (128, 128, 3) | 0 |
| **Conv2D** | 32 | 3 × 3 | ReLU | (126, 126, 32) | 896 |
| **MaxPooling2D**| — | 2 × 2 | — | (63, 63, 32) | 0 |
| **Conv2D** | 64 | 3 × 3 | ReLU | (61, 61, 64) | 18,496 |
| **MaxPooling2D**| — | 2 × 2 | — | (30, 30, 64) | 0 |
| **Conv2D** | 128 | 3 × 3 | ReLU | (28, 28, 128) | 73,856 |
| **MaxPooling2D**| — | 2 × 2 | — | (14, 14, 128) | 0 |
| **Flatten** | — | — | — | (25,088) | 0 |
| **Dense** | 256 | — | ReLU | (256) | 6,422,784 |
| **Dropout** | rate = 0.5 | — | — | (256) | 0 |
| **Dense** | 5 | — | Softmax | (5) | 1,285 |
| **Total Parameters** | **13,872,707** (Trainable: 13,872,707 / Non-trainable: 0) | | | | |

---

### 3.2 Mathematical Formulation

#### 1. Two-Dimensional Discrete Convolution
For input feature map $X$ and kernel weight matrix $K$ of size $M \times N$:

$$
S(i, j) = (X * K)(i, j) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X(i - m, j - n) \cdot K(m, n) + b
$$

#### 2. Rectified Linear Unit (ReLU) Activation
Introduces non-linearity to capture complex leaf venation geometry:

$$
f(z) = \max(0, z)
$$

#### 3. Spatial Max Pooling
Downsamples spatial dimensions while preserving translational invariance:

$$
P(i, j) = \max_{(m, n) \in \Omega} X(2i + m, 2j + n), \quad \Omega = \{0, 1\} \times \{0, 1\}
$$

#### 4. Inverted Dropout Regularization
Prevents co-adaptation of neurons during training by randomly zeroing activations with probability $p = 0.5$:

$$
\widetilde{y} = \frac{1}{1 - p} (y \odot m), \quad m_i \sim \text{Bernoulli}(1 - p)
$$

#### 5. Softmax Multi-Class Probability Distribution
Converts raw logit scores $z_i$ into a normalized probability distribution over the 5 botanical classes:

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{5} e^{z_j}}, \quad \sum_{i=1}^{5} \sigma(z)_i = 1.0
$$

#### 6. Categorical Cross-Entropy Loss Objective

$$
\mathcal{L}_{\text{CE}} = - \sum_{i=1}^{5} y_i \log(\hat{y}_i)
$$

where $y_i \in \{0, 1\}$ is the one-hot ground truth label and $\hat{y}_i$ is the predicted probability.

---

## 4. Repository Structure

```
Leaf/
├── .venv/                      # Isolated Python Virtual Environment
├── Lemon/                      # Lemon dataset images
├── Mango/                      # Mango dataset images
├── money/                      # Money Plant dataset images
├── Rosa-Senensis/              # Hibiscus / Rosa-senensis dataset images
├── Tulsi/                      # Holy Basil / Tulsi dataset images
├── leaf_model.h5               # Pretrained Custom CNN weights (~78 MB)
├── predict.py                  # CLI inference script for single/batch leaf predictions
├── train.py                    # Complete model training script with class balancing
├── evaluate.py                 # Evaluation script (confusion matrix, precision/recall/F1)
├── requirements.txt            # Pinned Python package dependencies
├── leaf_classification.ipynb   # Google Colab cloud execution notebook
├── main.pdf                    # Research paper preprint
└── README.md                   # College project documentation
```

---

## 5. Local Execution Guide (Virtual Environment)

### Step 1: Create the Virtual Environment
Open a terminal in the project directory:

```bash
# Windows PowerShell / CMD
python -m venv .venv
```

### Step 2: Activate the Virtual Environment

- **Windows PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If execution policies block scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

- **Windows Command Prompt (cmd)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Required Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Inference on a Leaf Image (`predict.py`)

To classify an individual leaf image:
```bash
python predict.py --image Lemon/1.jpg
```

**Output Preview:**
```text
======================================================================
           INDIAN HERBAL LEAF CLASSIFICATION REPORT
======================================================================
Image File      : Lemon/1.jpg
Predicted Leaf  : Lemon (Citrus limon)
Confidence Score: 98.42%
----------------------------------------------------------------------
Probability Distribution:
  • Lemon (Citrus limon)                       :  98.42% | ████████████████████████ 
  • Mango (Mangifera indica)                   :   0.84% |                           
  • Hibiscus / China Rose (Hibiscus rosa-sinensis) :   0.31% |                           
  • Holy Basil / Tulsi (Ocimum tenuiflorum)    :   0.25% |                           
  • Money Plant (Epipremnum aureum)            :   0.18% |                           
----------------------------------------------------------------------
Scientific Name : Citrus limon
Botanical Family: Rutaceae
Medicinal Uses  : Rich in Vitamin C and citric acid. Promotes digestion,
                  boosts immunity, and possesses antimicrobial properties.
======================================================================
```

To classify an entire folder of images in batch mode:
```bash
python predict.py --dir Lemon/
```

### Step 5: Evaluate the Model (`evaluate.py`)

Calculates test accuracy, classification metrics, and generates a confusion matrix heatmap:
```bash
python evaluate.py
```
This produces:
- Overall test accuracy percentage.
- Per-class Precision, Recall, and F1-Score table.
- A high-resolution confusion matrix saved as `confusion_matrix.png`.

### Step 6: Retrain or Fine-Tune the Model (`train.py`)

To retrain the custom CNN model from scratch with class weighting and data augmentation:
```bash
python train.py --epochs 20 --batch-size 32
```

For a quick 1-epoch pipeline check:
```bash
python train.py --dry-run
```
Outputs:
- Checkpoints saved to `leaf_model.h5`.
- Epoch loss & accuracy progression curves saved as `training_history.png`.

---

## 6. Cloud Execution Guide (Google Colab)

For fast GPU training without local setup:

1. Visit [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `leaf_classification.ipynb`.
3. Enable GPU acceleration:
   - Navigate to **Runtime** > **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** and click **Save**.
4. Upload your dataset:
   - Mount Google Drive or upload a `Leaf.zip` containing the class folders directly to Colab.
5. Run all cells (`Ctrl + F9` / `Cmd + F9`).
6. The interactive prediction cell at the bottom of the notebook allows you to upload any test leaf image and view live predictions and botanical data.

---

## 7. Model Performance & Experimental Results

- **Test Set Accuracy**: **91.1%** (achieved after 15–20 epochs)
- **Categorical Cross-Entropy Loss**: **0.21**
- **Convergence Behavior**: The custom CNN exhibits smooth loss decay with Adam optimization ($\eta = 0.001$), stabilizing around epoch 10. Dropout ($p=0.5$) prevents validation divergence and maintains tight generalization bounds.

---

## 8. Academic Attribution & Citation

This implementation is developed in accordance with the architectural specifications in:

```bibtex
@article{patel2025leaf,
  title={Leaf Classification Using A Custom CNN},
  author={Patel, Kishan and Shinde, Jayesh and Patel, Meghraj and Dhimmar, Jash},
  journal={Asha M. Tarsadia Institute of Computer Science and Technology Preprint},
  year={2025}
}
```
