import os
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix


def extract_path_and_label(dir_path):
    image_list = []
    image_label = []
    test_dir = os.listdir('data2_split/test2')
    for i in test_dir:
        # extract image path and append in the image_path list
        img = dir_path + '/' + i
        image_list.append(img)
        # extract actual label and append in the image_label list
        label = i.split(' ')
        image_label.append(label[0])
    return image_list, image_label


def give_prediction(test_path):
    model = YOLO("runs/classify/train6-adamw200-n001/weights/last.pt")
    prediction_list = []
    for img in test_path:
        results = model.predict(img)
        for r in results:
            prediction_idx = r.probs.top1
            if prediction_idx == 0:
                prediction_label = 'c1'
            elif prediction_idx == 1:
                prediction_label = 'c2'
            else:
                prediction_label = 'c3'
            prediction_list.append(str(prediction_label))
    return prediction_list


img_list, img_label = extract_path_and_label('data2_split/test2')
predict_list = give_prediction(img_list)
class_names = ['candidate 1', 'candidate 2', 'candidate 3']

print(f'List of Test Images: {img_list}')
print(f'List of Actual Label: {img_label}')
print(f'List of Prediction: {predict_list}')

print('\n\n')       # add spaces

print(f'Metric Evaluation - Multiclass [candidate 1, candidate 2, candidate 3]')     # evaluation metrics
print(f'{classification_report(img_label, predict_list, target_names=class_names)}')

print(f'Confusion Matrix')      # confusion matrix
print(f'{confusion_matrix(img_label, predict_list)}')
