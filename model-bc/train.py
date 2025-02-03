from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data='data1_split',
            epochs=200,
            imgsz=224,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.001,
            name='train6-adamw200-n001')    # optional

# split ratio 0.8, 0.2

# 1: SGD model-n lr 0.01 epochs 50 | 71.4% acc, 0.58436 train loss, 0.60628 val loss
# 2: SGD model-n lr 0.01 epochs 100 | 80.6% acc, 0.55265 train loss, 0.58152 val loss
# 3: SGD model-n lr 0.01 epochs 200 | 83.7% acc, 0.47275 train loss, 0.48202 val loss

# 4: AdamW model-n lr 0.001 epochs 50 | 72.5% acc, 0.54646 train loss, 0.60666 val loss
# 5: AdamW model-n lr 0.001 epochs 100 | 80.6% acc, 0.53278 train loss, 0.57943 val loss [BEST MODEL]
# 6: AdamW model-n lr 0.001 epochs 200 | 82.7% acc, 0.4384 train loss, 0.48664 val loss
