# Ballot Classifier

This final project is proposed to make a ballot classification system that give prediction from the input ballot image, based on presidential Indonesian general election 2024.
The system is a 2-staged image classification model using YOLOv8 algorithm, especially 'yolov8n-cls.pt'. <br>
- Model 1, named BCM (Binary Classification Model), classifies 2 classes = ['invalid', 'valid'] <br>
- Model 2, named MCM (Multiclass Classification Model), classifies 3 classes = ['candidate 1', 'candidate 2', 'candidate 3']

# Dataset
Dataset has 2 types of images: artificial images and printed images.
Both types are using the same ballot template illustrated by [Detik](https://news.detik.com/pemilu/d-7062623/ini-desain-resmi-surat-suara-pilpres-2024-segera-didistribusikan).
- Dataset for model 1 consists of 490 artificial images and 24 printed images.
- Dataset for model 2 consists of 240 artificial images and 12 printed images.

![image](https://github.com/user-attachments/assets/ebce8e93-a894-45c5-b55a-0a064f7ba391)

# System Design
![image](https://github.com/user-attachments/assets/0ff9557a-be61-4f2a-b175-1d53db894cdf)

# Model Loss & Accuracy (Model Training)
- BCM | Train accuracy 80.6% | Train loss 0.533 | Val loss 0.579
- MCM | Train accuracy 100.0% | Train loss 0.359 | Val loss 0.552

# Performance Evaluation (Model Testing)
- BCM | Test accuracy 95.8%
- MCM | Test accuracy 91.7%
