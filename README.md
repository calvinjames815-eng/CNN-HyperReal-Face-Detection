# CNN-HyperReal-Face-Detection

This project implements a lightweight deep learning pipeline using **MobileNetV2** for classifying real human faces and hyperreal AI-generated faces. The system applies a two-phase training strategy (transfer learning and fine-tuning), evaluates robustness using Test-Time Augmentation (TTA), and provides interpretability through Grad-CAM visualizations.

---

## Overview

The rapid advancement of Generative Adversarial Networks (GANs) has significantly reduced the visual differences between real and synthetic facial images. This project explores a computationally efficient approach for detecting such synthetic content using deep learning techniques suitable for resource-constrained environments.

The system is built on the **MobileNetV2 architecture pre-trained on ImageNet**, adapted using transfer learning and fine-tuning to classify real vs AI-generated faces. Explainability is integrated through Gradient-weighted Class Activation Mapping (Grad-CAM), enabling visualization of model decision patterns.

---

## Live Demo

A working prototype is available via Hugging Face Spaces:

https://huggingface.co/spaces/Redkahaja/hyperrealface-demo

---

## Dataset and Methodology

An **Ensemble Synthetic Dataset** was used to evaluate model robustness:

- **Real class (n = 200):** sourced from the CelebA dataset and other real-world face datasets
- **Synthetic class (n = 200):** generated using multiple GAN architectures including ThisPersonDoesNotExist and additional generative models

All images were:
- resized to **224 × 224 pixels**
- normalized to [0,1]
- augmented using rotation, flipping, and zoom transformations

A stratified dataset split was applied for training, validation, and testing to ensure balanced evaluation.

---

## Model Architecture

The system is based on **MobileNetV2**, configured as follows:

- Pre-trained ImageNet backbone (frozen initially)
- Global Average Pooling layer
- Fully connected Dense classification layer (Softmax output)
- Dropout regularization (0.2)

Training follows a two-stage process:
1. Transfer learning (frozen base layers)
2. Fine-tuning (last layers unfrozen with reduced learning rate)

---

## Evaluation and Robustness

The model was evaluated using:
- Classification Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC-AUC (reported in analysis)
- Test-Time Augmentation (TTA)

The system achieved a **held-out performance of approximately 91.25%**, indicating strong classification capability on the dataset.

TTA was applied to assess robustness under small perturbations such as flipping and brightness variation.

---

## Explainability (Grad-CAM)

Grad-CAM visualizations show that the model primarily focuses on:

- boundary regions between face and background
- hairline and jawline transitions
- texture inconsistencies in synthetic images

These regions correspond to **discriminative visual patterns** that help separate real and synthetic faces.

---

## Technical Limitations

While the model performs well under controlled conditions, several limitations were observed:

- **Domain Shift:** performance varies when applied to real-world mobile images with different lighting and sensor noise
- **Shortcut Learning:** the model may rely on background-related features rather than purely facial structure
- **Dataset Constraint:** limited dataset size (400 images) may restrict generalization to unseen generative models

---

## Conclusion

This project demonstrates that lightweight convolutional neural networks, particularly MobileNetV2, can effectively distinguish between real and AI-generated facial images under controlled conditions. The results suggest potential applications in digital forensics and misinformation detection, while also highlighting the need for more diverse and real-world datasets for improved generalization.

---

## Future Work

- Expansion to diffusion-based generative models
- Integration of larger in-the-wild datasets
- Reduction of shortcut learning via architectural constraints
- Improved robustness against adversarial manipulation

---

## Author

Calvin James B. Demegillo
