# Billboard Segmentation Using Segformer

This project leverages the Segformer pretrained model for billboard segmentation via semantic segmentation, with a focus on Indian billboards. By utilizing transfer learning on a custom dataset, the model precisely classifies and outlines billboards within images, facilitating efficient detection and analysis. The methodology incorporates several key improvements, including:

- **Dataset Augmentation** to increase model robustness.
- **Data Balancing** to address class imbalance issues.
- **Advanced Post-Processing** techniques to refine segmentation results.

These techniques significantly enhance segmentation accuracy, addressing challenges commonly encountered in binary semantic segmentation tasks.

---

## Dataset Format

The dataset is structured as follows:

```plaintext
├── Data
│   ├── Train
│   │   ├── images  # Training images
│   │   ├── masks   # Corresponding binary masks
│   ├── Validation
│   │   ├── images  # Validation images
│   │   ├── masks   # Corresponding binary masks
```

You can download the dataset from the following link:
- [Billboard Dataset](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/tree/main/Dataset)

---

## Google Colab Demo

To train the Segformer model for billboard segmentation, you can follow along with the Google Colab notebook linked below. The notebook provides step-by-step instructions for data preprocessing, model training, and evaluation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepchavda007/Billboard-Segmentation-using-Segformer/blob/main/Billboard_Segmentation.ipynb)

---

## Evaluation

The following graph shows the training and validation performance over time, demonstrating the model's convergence and improvement in segmentation accuracy:

![train_eval_plot_segformer-5-b1](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/assets/82630272/46913fdc-154a-45f5-8090-e8d2858dfde4)

---

## Test Images

Below are sample outputs of the model on test images. These showcase the model's ability to accurately segment billboards:

![test_4](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/assets/82630272/c240f459-f01a-4160-81e3-f3e0de64591c)

---

## Final Results

Here are some final outputs after post-processing, showing the refined billboard segmentation:

![1](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/assets/82630272/7c9a5ec5-ebe1-40fd-a855-f5e212b3f7b9)
![final_image_1](https://github.com/Deepchavda007/Billboard-Segmentation-using-Segformer/assets/82630272/13a6358c-bbcf-47df-bfdf-d048140e34b1)

---

This README is focused solely on the training aspect of the project. A separate README file will cover API-related details and deployment.

---
