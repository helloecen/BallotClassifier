from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data='data2_split',
            epochs=200,
            imgsz=224,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.001,
            name='train6-adamw200-n001')    # optional

# split ratio 0.8, 0.2

# 1: SGD model-n lr 0.01 epochs 50 | 81.2% acc, 0.62523 train loss, 0.77939 val loss
# 2: SGD model-n lr 0.01 epochs 100 | 100.0% acc, 0.35243 train loss, 0.57419 val loss
# 3: SGD model-n lr 0.01 epochs 200 | 100.0% acc, 0.33925 train loss, 0.55264 val loss

# 4: AdamW model-n lr 0.001 epochs 50 | 79.2% acc, 0.60150 train loss, 0.78668 val loss
# 5: AdamW model-n lr 0.001 epochs 100 | 100.0% acc, 0.42169 train loss, 0.56972 val loss
# 6: AdamW model-n lr 0.001 epochs 200 | 100.0% acc, 0.35857 train loss, 0.55211 val loss [BEST MODEL]
