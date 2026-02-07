# 🧠 Shoe Image Classification – CNN from Scratch

## 📌 Description

This project implements a **Convolutional Neural Network (CNN)** for **shoe image classification**. The goal is to train a model **from scratch** (without pre-trained models) in order to understand:

* CNN architecture design,
* data augmentation techniques,
* the training loop,
* evaluation and prediction on real images.

> **Performance note:** the achieved accuracy is around **64%**. This is mainly due to the **small size of the dataset** and the limited diversity of training samples.

## 📂 Dataset & Data Collection

The dataset is organized using a folder-based structure (1 folder = 1 class):

```
shose/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── test/
│   ├── class_1/
│   ├── class_2/
│   └── ...
```

### 🧾 Data Collection

The dataset was built by collecting images (from the web / public sources) representing different shoe brands and styles.

Examples of brands included in the dataset:

* **Nike**
* **Adidas**
* **Converse**

> The data collection process and image quality (view angles, lighting, background, resolution) have a strong impact on the model’s performance.

## 🛠️ Technologies Used

* Python 3
* PyTorch + Torchvision
* NumPy
* PIL (Pillow)
* Matplotlib
* (optional) TensorFlow / Keras

## 🔄 Preprocessing & Data Augmentation

### Training

* `Resize` → 256×256
* `RandomResizedCrop` → 224×224 (scale 0.85–1.0)
* `RandomHorizontalFlip` (p = 0.5)
* `ColorJitter` (brightness, contrast, saturation)
* `ToTensor`

### Testing

* `Resize` → 224×224
* `ToTensor`

## 🧱 Model Architecture (SimpleCNN)

```
Input (3 × 224 × 224)
│
├─ Conv2D (16) → ReLU → MaxPool
├─ Conv2D (32) → ReLU → MaxPool
├─ Conv2D (64) → ReLU → AdaptiveAvgPool(1×1)
│
├─ Flatten
├─ Dense 128 → ReLU
└─ Dense N_CLASSES (logits)
```

* **Loss function**: CrossEntropyLoss
* **Optimizer**: Adam (lr = 0.001)

## 🚀 Training

The model is trained for **20 epochs** with a **batch size of 64**.

Command:

```bash
python CNN_PyTorch.py
```

## 📊 Evaluation

After each epoch:

* average loss is computed
* **accuracy** is evaluated on the test dataset

> Observed result: **~64% accuracy** (small dataset + limited diversity).

## 💾 Model Saving & Loading

```python
torch.save(model.state_dict(), "shos.pth")
model.load_state_dict(torch.load("shos.pth"))
```

## 🖼️ Image Prediction

```python
img = Image.open("45e62.jpg")
output = model(img.unsqueeze(0))
pred = output.argmax(dim=1).item()
```

## 🔜 Possible Improvements

* Increase dataset size (more images per class)
* Data cleaning (remove duplicates, blurry or low-quality images)
* Add **Batch Normalization** and **Dropout** layers
* Apply **Transfer Learning** (ResNet, MobileNet)
* Plot loss/accuracy curves and confusion matrix
* Export model to ONNX / TensorFlow Lite

## 👨‍🎓 Author

Project developed by **Kevin** as part of learning **Deep Learning with CNNs**.
