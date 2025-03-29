# Osteoporosis Classification using Deep Learning

## Project Overview
This project focuses on classifying bone conditions using deep learning models trained on X-ray images. The models used include **VGG16, VGG19, InceptionV3, ResNet50, Xception, AlexNet, and a Custom CNN**. The goal is to accurately classify images into three categories:

- **Osteopenia**
- **Osteoporosis**
- **Normal**

## Dataset
The dataset consists of **X-ray images** of bones, divided into three classes. The images were preprocessed by resizing, normalizing, and augmenting to enhance the model's performance.

## Models Used
We have trained and evaluated the following deep learning models:
1. **VGG16**
2. **VGG19**
3. **InceptionV3**
4. **ResNet50**
5. **Xception**
6. **AlexNet**
7. **Custom CNN**

Each model was trained with the same dataset and evaluated using precision, recall, f1-score, accuracy, and confusion matrices.

## Performance Metrics
Below is a summary of the classification performance for each model:

| Model       | Accuracy | Precision | Recall | F1-Score | Confusion Matrix | Graphs |
|------------|----------|------------|--------|----------|------------------|--------|
| **VGG16**  | 70%      | 0.74       | 0.70   | 0.70     | ✅ | ✅ |
| **VGG19**  | 75%      | 0.77       | 0.75   | 0.75     | ✅ | ✅ |
| **InceptionV3** | 88% | 0.89       | 0.88   | 0.88     | ✅ | ✅ |
| **ResNet50** | 66%   | 0.74       | 0.66   | 0.65     | ✅ | ✅ |
| **Xception** | 87%   | 0.88       | 0.87   | 0.87     | ✅ | ✅ |
| **AlexNet** | 85%    | 0.86       | 0.85   | 0.85     | ✅ | ✅ |
| **Custom CNN** | 89% | 0.89       | 0.89   | 0.89     | ✅ | ✅ |

## Classification Reports
### **VGG16**
```
              precision    recall  f1-score   support

  Osteopenia       0.82      0.55      0.66        75
Osteoporosis       0.60      0.86      0.71       159
      Normal       0.83      0.60      0.70       156

    accuracy                           0.70       390
   macro avg       0.75      0.67      0.69       390
weighted avg       0.74      0.70      0.70       390
```

### **VGG19**
```
              precision    recall  f1-score   support

  Osteopenia       0.81      0.73      0.77        75
Osteoporosis       0.68      0.88      0.77       159
      Normal       0.85      0.63      0.73       156

    accuracy                           0.75       390
   macro avg       0.78      0.75      0.75       390
weighted avg       0.77      0.75      0.75       390
```

### **InceptionV3**
```
              precision    recall  f1-score   support

  Osteopenia       0.85      0.88      0.86        75
Osteoporosis       0.88      0.91      0.89       159
      Normal       0.91      0.87      0.89       156

    accuracy                           0.88       390
   macro avg       0.88      0.88      0.88       390
weighted avg       0.89      0.88      0.88       390
```

### **ResNet50**
```
              precision    recall  f1-score   support

  Osteopenia       0.87      0.35      0.50        75
Osteoporosis       0.57      0.92      0.70       159
      Normal       0.86      0.55      0.67       156

    accuracy                           0.66       390
   macro avg       0.76      0.61      0.62       390
weighted avg       0.74      0.66      0.65       390
```

### **Xception**
```
              precision    recall  f1-score   support

  Osteopenia       0.81      0.89      0.85        75
Osteoporosis       0.89      0.83      0.86       159
      Normal       0.89      0.91      0.90       156

    accuracy                           0.87       390
   macro avg       0.86      0.88      0.87       390
weighted avg       0.88      0.87      0.87       390
```

### **AlexNet**
```
              precision    recall  f1-score   support

  Osteopenia       0.86      0.85      0.86        75
Osteoporosis       0.82      0.88      0.85       159
      Normal       0.89      0.83      0.86       156

    accuracy                           0.85       390
   macro avg       0.86      0.85      0.85       390
weighted avg       0.86      0.85      0.85       390
```

### **Custom CNN**
```
              precision    recall  f1-score   support

  Osteopenia       0.82      0.75      0.78        75
Osteoporosis       0.87      0.92      0.90       159
      Normal       0.94      0.92      0.93       156

    accuracy                           0.89       390
   macro avg       0.88      0.86      0.87       390
weighted avg       0.89      0.89      0.89       390
```

## Confusion Matrices & Graphs
Each model has an associated **confusion matrix** and **performance graphs** showcasing:
- **Training & Validation Accuracy**
- **Training & Validation Loss**
- **Comparative Model Performance**

## Conclusion
Among all models, **Custom CNN** performed the best with **89% accuracy**, followed by **InceptionV3** at **88%**. The **VGG and ResNet architectures** showed moderate performance. The **confusion matrices and graphs** provide further insights into model performance.

## Authors
- **[Your Name]** - AI/ML Researcher

---
*This project was developed as part of an ongoing research initiative in medical image classification using deep learning.*

