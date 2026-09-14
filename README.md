# 🌸 Flower Classifier

A flower image classifier built with **Transfer Learning** on top of **MobileNetV2**, fine-tuned to recognize 5 types of flowers. The model was trained in Google Colab and is used here locally with a simple image-picker interface powered by OpenCV.

## ✨ Features

- Classifies an image into one of 5 flower types: daisy, dandelion, roses, sunflowers, tulips
- Built on **MobileNetV2**, a lightweight model pre-trained on ImageNet
- Uses both **Feature Extraction** and **Fine-Tuning** stages for better accuracy
- Simple file-picker interface — select any flower photo from your device and get an instant prediction

## 🧠 How This Model Was Built

This project was built in two main stages: training in Google Colab, and inference locally with OpenCV.

### Stage 1: Training in Google Colab

1. The **flower_photos** dataset (3,670 images across 5 classes: daisy, dandelion, roses, sunflowers, tulips) was downloaded directly using `tf.keras.utils.get_file`
2. Images were loaded and split into training/validation sets using `image_dataset_from_directory`
3. **MobileNetV2** was loaded with pre-trained ImageNet weights, with its top classification layer removed (`include_top=False`)
4. A new classification head was added on top: Global Average Pooling → Dropout → Dense (5 classes, softmax)
5. **Stage A - Feature Extraction:** The MobileNetV2 base was frozen (`base_model.trainable = False`), and only the new head was trained for 10 epochs. This reached about **89% validation accuracy**
6. **Stage B - Fine-Tuning:** The base model was unfrozen, but the first 100 layers were kept frozen (since early layers learn generic features like edges and textures that don't need to change). The remaining deeper layers were retrained with a much smaller learning rate (0.00001) for 10 more epochs, to carefully adjust the pre-trained features toward this specific flower dataset without destroying what MobileNetV2 already learned from ImageNet
7. The final trained model was exported as `flower_classifier.keras`, along with `class_names.json` containing the label names

### Stage 2: Local Inference with OpenCV

- The saved model and class names are loaded locally
- **OpenCV** and **Tkinter** are used to open a native file picker, letting the user select any flower image from their device
- The selected image is resized, preprocessed the same way MobileNetV2 expects, and passed through the model
- The predicted flower type and confidence score are displayed directly on the image in a popup window

## 📊 Model Performance

| Stage | Validation Accuracy |
|-------|---------------------|
| Feature Extraction (base frozen) | ~89.4% |
| After Fine-Tuning (partial unfreeze) | ~89.6% |

## 📦 Requirements

- Python 3.9 or newer
- No GPU required for inference (runs on CPU)

## 📥 Clone the Repository

```bash
git clone https://github.com/kamandNajari/Flower_Classifier.git
cd Flower_Classifier
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📓 Running the Notebook

Make sure Jupyter is installed:

```bash
pip install jupyter
```

Then launch it:

```bash
jupyter notebook
```

Open `flower_classifier_inference.ipynb` from the Jupyter interface and run the cell (Shift + Enter).

## ▶️ How to Use

1. Run the notebook cell — a file picker window will open
2. Select any flower image from your device (`.jpg`, `.jpeg`, `.png`, or `.bmp`)
3. A window pops up showing the image with the predicted flower type and confidence percentage

## 🛠️ Tech Stack

- [TensorFlow / Keras](https://www.tensorflow.org/) — MobileNetV2, transfer learning, and fine-tuning
- [OpenCV](https://opencv.org/) — image loading, processing, and display
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — native file picker dialog

## 🔮 Planned Improvements

The current model only recognizes 5 flower types due to the limited dataset used. A future version is planned using the **Oxford 102 Flower Dataset**, which will expand recognition to over 100 different flower species using the same MobileNetV2 transfer learning approach.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

Thank you for checking out this project! ✨
