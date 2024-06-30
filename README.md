# Final Project - Ballot Classifier | Jason Felix

This final project is planned to make a ballot classification system to give prediction from the input ballot image, based on presidential Indonesian general election 2024.
The proposed system is a 2-staged image classification model using YOLOv8 architecture, especially 'yolov8n-cls.pt'. <br>
- Model 1 classifies 2 classes = ['invalid', 'valid'] <br>
- Model 2 classifies 3 classes = ['candidate 1', 'candidate 2', 'candidate 3']

# Dataset
Dataset has 2 types of images: artificial images (made using Canva) and printed images.
Both types are using the same ballot template illustrated by Detik ([link](https://news.detik.com/pemilu/d-7062623/ini-desain-resmi-surat-suara-pilpres-2024-segera-didistribusikan)).
- Dataset for model 1 consists of 370 artificial images and 21 printed images.
- Dataset for model 2 consists of 240 artificial images and 9 printed images. (images from first model's valid class)

# System Design
![systemdesign](https://github.com/helloecen/FinalProject-BallotClassifier/assets/157099933/84ada52f-0737-4635-8673-339a1921bb85)

# Model Loss & Accuracy
- Model 1 | Train top1 accuracy 100.0% | Train loss 0.143 | Val loss 0.320
- Model 2 | Train top1 accuracy 100.0% | Train loss 0.356 | Val loss 0.559

# Performance Evaluation
- Model 1 | Test accuracy 100.0%
- Model 2 | Test accuracy 92.0%
