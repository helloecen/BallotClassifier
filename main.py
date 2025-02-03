import cv2

from ultralytics import YOLO

# weights from trained model
model1 = YOLO("model-bc/runs/classify/train5-adamw100-n001/weights/last.pt")     # BCM
model2 = YOLO("model-mc/runs/classify/train6-adamw200-n001/weights/last.pt")     # MCM

def get_prediction_1(img):
    results = model1.predict(img)
    for r in results:
        predict_idx = r.probs.top1          # top 1 prediction
        conf = r.probs.top1conf.item()      # confidence probability
        if predict_idx == 0:
            label = 'INVALID'
        elif predict_idx == 1:
            label = 'VALID'
    return label, conf

def get_prediction_2(img):
    results = model2.predict(img)
    for r in results:
        predict_idx = r.probs.top1          # top1 prediction
        conf = r.probs.top1conf.item()      # confidence probability
        if predict_idx == 0:
            label = 'Candidate 1'
        elif predict_idx == 1:
            label = 'Candidate 2'
        elif predict_idx == 2:
            label = 'Candidate 3'
    return label, conf


def display_output(img_path):
    # image read and resize window
    image = cv2.imread(img_path)
    image_resized = cv2.resize(image, (900, 900))

    # prediction label and confidence probs
    label1, conf1 = get_prediction_1(img_path)
    label2, conf2 = get_prediction_2(img_path)

    # text background
    cv2.rectangle(image_resized, (5, 855), (895, 895), (255, 255, 255), -1)

    # prediction text
    if label1 == 'VALID':
        text_valid = f'BC = {label1} {float(conf1) * 100:.2f}%, MC = {label2} {float(conf2) * 100:.2f}%'
        cv2.putText(image_resized, text_valid, (10, 885), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        text_invalid = f'BC = {label1} {float(conf1) * 100:.2f}%'
        cv2.putText(image_resized, text_invalid, (10, 885), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Ballot Classifier', image_resized)
    # cv2.imwrite('system-result/output-inv12.jpg', image_resized)      # save the output
    cv2.waitKey(0)

image_path = 'model-bc/data1_split/test1/invalid (12).jpg'
display_output(image_path)
