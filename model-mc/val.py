from ultralytics import YOLO

model = YOLO('runs/classify/train1-sgd50-n01/weights/last.pt')

model.val()
