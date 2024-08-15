# Final Project - Ballot Classifier | Jason Felix

This final project is planned to make a ballot classification system to give prediction from the input ballot image, based on presidential Indonesian general election 2024.
The proposed system is a 2-staged image classification model using YOLOv8 architecture, especially 'yolov8n-cls.pt'. <br>
- Model 1 classifies 2 classes = ['invalid', 'valid'] <br>
- Model 2 classifies 3 classes = ['candidate 1', 'candidate 2', 'candidate 3']

# Dataset
Dataset has 2 types of images: artificial images (made using Canva) and printed images.
Both types are using the same ballot template illustrated by Detik ([link](https://news.detik.com/pemilu/d-7062623/ini-desain-resmi-surat-suara-pilpres-2024-segera-didistribusikan)).
- Dataset for model 1 consists of 490 artificial images and 24 printed images.
- Dataset for model 2 consists of 240 artificial images and 12 printed images. (images from first model's valid class)
![image](https://github.com/user-attachments/assets/0727d97c-0e0a-4375-a3a7-e585b61537cb)


# System Design
![image](https://github.com/user-attachments/assets/3391528b-e2e8-48ca-b7c9-9464ae65270d)


# Model Loss & Accuracy
- Model 1 | Train accuracy 80.6% | Train loss 0.533 | Val loss 0.579
- Model 2 | Train accuracy 100.0% | Train loss 0.359 | Val loss 0.552

# Performance Evaluation
- Model 1 | Test accuracy 95.8%
- Model 2 | Test accuracy 91.7%
