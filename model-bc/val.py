from ultralytics import YOLO

model = YOLO('runs/classify/train6-adamw200-n001/weights/last.pt')

model.val()
