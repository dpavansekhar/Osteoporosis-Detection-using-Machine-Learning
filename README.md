# Osteoporosis Classification using Deep Learning

## Project Overview
This project focuses on classifying bone conditions using deep learning models trained on X-ray images. The models used include **VGG16, VGG19, InceptionV3, ResNet50, Xception, AlexNet, and a Custom CNN**. The goal is to accurately classify images into three categories:

- **Osteopenia** ![image](https://github.com/user-attachments/assets/a2f76186-b93e-4638-8c2b-4d00a8803c99)
- **Osteoporosis** ![image](https://github.com/user-attachments/assets/9ae100e7-1ffd-4d6f-947e-076fa5fcf819)
- **Normal** ![image](https://github.com/user-attachments/assets/2780e25a-7f1f-492d-ade2-a75d449d2d4f)

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
| **VGG16**  | 70%      | 0.74       | 0.70   | 0.70     | ![image](https://github.com/user-attachments/assets/94092c92-4344-4a42-9e8a-ffae9e0ac671) | ![image](https://github.com/user-attachments/assets/7bb28717-0c57-44d5-ac54-a249e8840db5) |
| **VGG19**  | 75%      | 0.77       | 0.75   | 0.75     | ![image](https://github.com/user-attachments/assets/2c4da798-a558-45c6-965f-a7f4f48b1ba9) | ![image](https://github.com/user-attachments/assets/06218327-c539-4199-bd16-ebe6cd3f1969) |
| **InceptionV3** | 88% | 0.89       | 0.88   | 0.88     | ![image](https://github.com/user-attachments/assets/14f0fbe5-bda7-42fc-be21-b55f746739a8) | ![image](https://github.com/user-attachments/assets/af83ce3a-99f3-438a-a8b8-1b4b49f06c93) |
| **ResNet50** | 66%   | 0.74       | 0.66   | 0.65     | ![image](https://github.com/user-attachments/assets/f5fac108-bfa9-4571-b94a-4ff18437a09b) | ![image](https://github.com/user-attachments/assets/cd8b76ad-cea7-4158-a446-4048dcffdaf7) |
| **Xception** | 87%   | 0.88       | 0.87   | 0.87     | ![image](https://github.com/user-attachments/assets/4c5dba18-a3ab-4b58-920d-0bcbfd2053d6) | ![image](https://github.com/user-attachments/assets/9100cc15-e631-4e2a-bc15-5ebb7934e191) |
| **AlexNet** | 85%    | 0.86       | 0.85   | 0.85     | ![image](https://github.com/user-attachments/assets/25120dd6-fa30-412b-b3ef-6b3ae0ed6d6d) | ![image](https://github.com/user-attachments/assets/0ad3578c-8c1a-4f9f-9f37-23640e3211f3) |
| **Custom CNN** | 89% | 0.89       | 0.89   | 0.89     | ![image](https://github.com/user-attachments/assets/962d3275-17d7-4c9e-86ec-fdf57c72f504) | ![image](https://github.com/user-attachments/assets/1c5899e5-3d42-4aa9-84b3-7d4fedfd570c) |

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
- **[Dogga Pavan Sekhar]** - AI/ML Researcher

---
*This project was developed as part of an ongoing research initiative in medical image classification using deep learning.*

