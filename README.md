# 🧠 Brain Tumor Classification with Deep Learning

This project is a deep learning application that classifies brain MRI images into one of four categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. It uses a Convolutional Neural Network (CNN) trained on MRI images and features a user-friendly interface built with **Gradio**.

## 📁 Dataset

The dataset is expected to be structured as follows (in Google Drive):


Brain Tumor Segmentation/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/

Each subdirectory contains brain MRI images for the respective class.

## 🚀 Features

- Image loading and preprocessing with OpenCV
- CNN architecture with Conv2D, MaxPooling, BatchNorm, and Dropout
- One-hot encoding of labels using `LabelBinarizer`
- Train-test split and model evaluation
- Deployment with Gradio for live inference

## 🏗️ Model Architecture

```text
Input (128x128x3)
→ Conv2D + ReLU + MaxPool + BatchNorm
→ Conv2D + ReLU + MaxPool + BatchNorm
→ Conv2D + ReLU + MaxPool + BatchNorm
→ GlobalAveragePooling
→ Dense + Dropout
→ Output Layer (Softmax over 4 classes)
````

## 📦 Dependencies

Install the required libraries:

```bash
pip install gradio opencv-python tensorflow numpy scikit-learn
```

## 🧪 Training

The model is trained with the following hyperparameters:

* Image size: 128x128
* Batch size: 16
* Epochs: 10
* Loss: Categorical Crossentropy
* Optimizer: Adam

## 🖼️ Live Demo

Once trained, a **Gradio** interface allows users to upload MRI images and receive predictions in real time:

```python
gr.Interface.launch()
```

## 📸 Example Prediction

Upload an image via the Gradio interface and receive a label:

```
Prediction: pituitary
```

## 🛠️ How to Run

1. Mount your Google Drive in Colab or set the dataset path locally.
2. Run all cells in `BrainTumor.ipynb`.
3. Launch the Gradio app and test your model with sample images.

## 📌 Notes

* Make sure all image data is pre-organized into class folders.
* The model normalizes pixel values to the range \[0, 1].

## 📄 License

This project is open-source under the [MIT License](LICENSE).

