import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

result = pd.read_csv('model-mc/runs/classify/train6-adamw200-n001/results.csv')

epoch = result['                  epoch']
tra_loss = result['             train/loss']
tra_smooth = gaussian_filter1d(tra_loss, sigma=2)
val_loss = result['               val/loss']
val_smooth = gaussian_filter1d(val_loss, sigma=2)
acc = result['  metrics/accuracy_top1']
acc_smooth = gaussian_filter1d(acc, sigma=2)

plt.figure()
plt.plot(epoch, tra_smooth, label='train loss')
plt.plot(epoch, val_smooth, label='val loss', c='red', ls='--')
plt.grid()
plt.ylabel('Loss', fontsize='x-large', fontweight='semibold')
plt.xlabel('Epochs', fontsize='x-large', fontweight='semibold')
plt.legend(fontsize='x-large', loc='upper right')

plt.figure()
plt.plot(epoch, acc_smooth*100, label='acc')
plt.grid()
plt.ylabel('Accuracy', fontsize='x-large', fontweight='semibold')
plt.xlabel('Epochs', fontsize='x-large', fontweight='semibold')
plt.legend(fontsize='x-large', loc='lower right')

plt.show()