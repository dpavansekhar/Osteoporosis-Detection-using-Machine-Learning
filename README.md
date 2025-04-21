# Osteoporosis Classification using Deep Learning
---
## Project Overview
This project focuses on classifying bone conditions using deep learning models trained on X-ray images. The models used include **VGG16, VGG19, InceptionV3, ResNet50, Xception, AlexNet, MobileNetV2 and a Custom CNN**. The goal is to accurately classify images into three categories:

- **Osteopenia** 
- **Osteoporosis** 
- **Normal** 
---
## Dataset
The dataset consists of **X-ray images** of bones, divided into three classes. The images were preprocessed by resizing, normalizing, and augmenting to enhance the model's performance.

|X-ray Images||||Classification|
|----------------------|----------------------|----------------------|----------------------|----------------------|
|![Normal](https://github.com/user-attachments/assets/bdbe54bf-a7f7-45ef-acaa-f54e38c6f6ae)|![Normal](https://github.com/user-attachments/assets/73aef2e4-d48e-4948-b681-39e68e7318c5)|![Normal](https://github.com/user-attachments/assets/92c02e3f-e4bb-4539-a8d4-a22d0190c783)|![Normal](https://github.com/user-attachments/assets/9aa1826a-3e18-482e-9e67-a623b3112462)|Normal|
|![Osteopenia](https://github.com/user-attachments/assets/58838b3b-7d38-478b-8cbd-1df131fc9baf)|![Osteopenia](https://github.com/user-attachments/assets/5bbfb755-1fef-4c97-a125-8f4f1a1428fb)|![Osteopenia](https://github.com/user-attachments/assets/894b6af5-99f4-45ce-9512-7bc04c9e5501)|![Osteopenia](https://github.com/user-attachments/assets/ad5a447d-d298-4cbb-97ff-658bb94bdd55)|Osteopenia|
|![Osteoporosis](https://github.com/user-attachments/assets/7456a0cc-38d0-4f14-a176-f22cdf0a5d63)|![Osteoporosis](https://github.com/user-attachments/assets/adf9ab78-dc4f-4ad7-8a25-cac85940d70a)|![Osteoporosis](https://github.com/user-attachments/assets/e8f2773a-d0a4-4753-a55a-0c434aebbea2)|![Osteoporosis](https://github.com/user-attachments/assets/a450f36f-aced-4f55-82a1-85250a0ccc71)|Osteoporosis|


## Model Architecture
![Model Architecture of Osteoporosis Prediction](https://github.com/user-attachments/assets/f7380181-ab19-41bf-b6c5-f159761a6057)
---
## Models Used
We have trained and evaluated the following deep learning models:
1. **VGG16**
2. **VGG19**
3. **InceptionV3**
4. **ResNet50**
5. **Xception**
6. **AlexNet**
7. **Custom CNN**
8. **Late Fusion**
9. **Dense Net 121**
10. **VGG 16 + VGG 19**
11. **InceptionV3 + XceptionNet**
12. **ResNet 50 + DenseNet 121**

Each model was trained with the same dataset and evaluated using precision, recall, f1-score, accuracy, and confusion matrices.
---
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
| **MobileNetV2** | 84% | 0.82 | 0.84 | 0.82 | ![image](https://github.com/user-attachments/assets/c7c04cfe-5f43-4d88-9806-7d32f4c4103c) | ![image](https://github.com/user-attachments/assets/6ef85407-5f80-45f7-aba7-c5fe5af90d25)| 
| **Custom CNN** | 89% | 0.89       | 0.89   | 0.89     | ![image](https://github.com/user-attachments/assets/962d3275-17d7-4c9e-86ec-fdf57c72f504) | ![image](https://github.com/user-attachments/assets/1c5899e5-3d42-4aa9-84b3-7d4fedfd570c) |
|*Ensemble Learning*|
|**VGG 16 + VGG 19**|80%|0.81|0.80|0.81|![image](https://github.com/user-attachments/assets/ab18b32f-fef4-43c9-a0c5-719a1d9ee694)||
|**InceptionV3 + Xception**|84%|0.83|0.84|0.83|![image](https://github.com/user-attachments/assets/0a26ff86-a516-45a9-943b-f4c0ddc231cd)||
|**ResNet50 + DenseNet121**|82%|0.82|0.78|0.79|![image](https://github.com/user-attachments/assets/178e1015-a7ed-42f2-809e-e156b15bd54c)||
|**AlexNet + MobileNetV2**|88%|0.88|0.87|0.88|||
|**InceptionV3 + DenseNet121**|85%|0.84|0.85|0.84|||
|**Xception + DenseNet121**|85%|0.84|0.85|0.84|||
|**MobileNetV2 + Xception**|84%|0.83|0.84|0.83|||
|**InceptionV3 + MobileNetV2**|84%|0.83|0.83|0.84|||
---
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

### **MobileNet V2**
```
              precision    recall  f1-score   support

  Osteopenia       0.73      0.72      0.72        75
Osteoporosis       0.75      0.83      0.79       159
      Normal       0.86      0.78      0.82       156

    accuracy                           0.79       390
   macro avg       0.78      0.78      0.78       390
weighted avg       0.79      0.79      0.79       390
```

## Ensemble Learning

### **VGG 16 + VGG 19**
```
              precision    recall  f1-score   support

  Osteopenia       0.78      0.83      0.80        75
Osteoporosis       0.75      0.86      0.80       159
      Normal       0.89      0.73      0.80       156

    accuracy                           0.80       390
   macro avg       0.81      0.81      0.80       390
weighted avg       0.81      0.80      0.80       390
```

### **InceptionV3 + Xception**
```
              precision    recall  f1-score   support

  Osteopenia       0.78      0.77      0.78        75
Osteoporosis       0.83      0.87      0.85       159
      Normal       0.89      0.85      0.87       156

    accuracy                           0.84       390
   macro avg       0.83      0.83      0.83       390
weighted avg       0.84      0.84      0.84       390
```

### **ResNet50 + DenseNet121**
```
              precision    recall  f1-score   support

  Osteopenia       0.83      0.57      0.68        75
Osteoporosis       0.79      0.92      0.85       159
      Normal       0.85      0.84      0.85       156

    accuracy                           0.82       390
   macro avg       0.82      0.78      0.79       390
weighted avg       0.82      0.82      0.82       390
```
### **AlexNet + MobileNetV2**
```
              precision    recall  f1-score   support

  Osteopenia       0.86      0.81      0.84        75
Osteoporosis       0.85      0.94      0.89       159
      Normal       0.94      0.85      0.89       156

    accuracy                           0.88       390
   macro avg       0.88      0.87      0.87       390
weighted avg       0.89      0.88      0.88       390
```
### **InceptionV3 + DenseNet121**
```
              precision    recall  f1-score   support

  Osteopenia       0.75      0.85      0.80        75
Osteoporosis       0.84      0.86      0.85       159
      Normal       0.93      0.84      0.88       156

    accuracy                           0.85       390
   macro avg       0.84      0.85      0.84       390
weighted avg       0.86      0.85      0.85       390
```
### **Xception + DenseNet121**
```
              precision    recall  f1-score   support

  Osteopenia       0.80      0.88      0.84        75
Osteoporosis       0.81      0.88      0.84       159
      Normal       0.93      0.79      0.86       156

    accuracy                           0.85       390
   macro avg       0.84      0.85      0.84       390
weighted avg       0.85      0.85      0.85       390
```
### **MobileNetV2 + Xception**
```
              precision    recall  f1-score   support

  Osteopenia       0.76      0.85      0.81        75
Osteoporosis       0.79      0.88      0.83       159
      Normal       0.95      0.78      0.86       156

    accuracy                           0.84       390
   macro avg       0.83      0.84      0.83       390
weighted avg       0.85      0.84      0.84       390
```
### **InceptionV3 + MobileNetV2**
```
              precision    recall  f1-score   support

  Osteopenia       0.72      0.80      0.76        75
Osteoporosis       0.82      0.86      0.84       159
      Normal       0.94      0.85      0.89       156

    accuracy                           0.84       390
   macro avg       0.83      0.83      0.83       390
weighted avg       0.85      0.84      0.84       390
```
---
## Confusion Matrices & Graphs
Each model has an associated **confusion matrix** and **performance graphs** showcasing:
- **Training & Validation Accuracy**
- **Training & Validation Loss**
- **Comparative Model Performance**
---
## Conclusion
Among all models, **Custom CNN** performed the best with **89% accuracy**, followed by **InceptionV3** at **88%**. The **VGG and ResNet architectures** showed moderate performance. The **confusion matrices and graphs** provide further insights into model performance.
---
## Authors
- **[Dogga Pavan Sekhar](https://www.linkedin.com/in/dogga-pavan-sekhar-006a83252/)** - AI/ML Researcher
---
*This project was developed as part of an ongoing research initiative in medical image classification using deep learning.*

