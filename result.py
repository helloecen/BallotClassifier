import matplotlib.pyplot as plt
import os

image_dir = "sys3"
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

fig, axes = plt.subplots(1, 2)

for i, image in enumerate(images):
    img = plt.imread(image)
    row, col = divmod(i, 2)
    axes[row, col].imshow(img)
    axes[row, col].axis('off')

# fig.suptitle("System Output", fontsize='xx-large', fontweight='bold')

plt.tight_layout()

plt.show()
