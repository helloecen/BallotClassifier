import os
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

def extract_path_and_label(dir_path):
    image_list = []
    image_label = []
    test_dir = os.listdir('mix_split/test')
    for i in test_dir:
        # extract image path and append in the image_path list
        img = dir_path + '/' + i
        image_list.append(img)
        # extract actual label and append in the image_label list
        label = i.split(' ')
        image_label.append(label[0])
    return image_list, image_label

def give_prediction(test_path):
    model = YOLO("runs/classify/train5-adamw100-n001/weights/last.pt")
    prediction_list = []
    for img in test_path:
        results = model.predict(img)
        for r in results:
            # r.probs.top1 to call the top prediction from the model
            prediction_idx = r.probs.top1
            if prediction_idx == 0:
                prediction_label = 'invalid'
            else:
                prediction_label = 'valid'
            prediction_list.append(str(prediction_label))
    return prediction_list

img_list, img_label = extract_path_and_label('mix_split/test')
# print(img_list)
# print(img_label)

predict_list = give_prediction(img_list)
class_names = ['invalid ballot', 'valid ballot']

print('\n\n')       # add spaces

print(f'Metric Evaluation - Binary [valid, invalid]')     # evaluation metrics
print(f'{classification_report(img_label, predict_list, target_names=class_names)}')

print(f'Confusion Matrix')      # confusion matrix
print(f'{confusion_matrix(img_label, predict_list)}')
