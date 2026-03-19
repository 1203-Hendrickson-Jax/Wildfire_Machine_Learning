import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

builder = __import__("dataset_creator").WildfireSmoke()
builder.download_and_prepare()
info = builder.info

ds_train = tfds.load("wildfire_smoke", split="train") # 70% split
ds_val = tfds.load("wildfire_smoke", split="validation") # 15% split
ds_test = tfds.load("wildfire_smoke", split="test") # 15% split

print(info)

def preprocess_image(example):
    image = tf.cast(example['image'], tf.float32)

    # Crop black bar from bottom and logo from top
    cropped_image = image[100:-25, :, :]

    # Resize
    resized_image = tf.image.resize(cropped_image, (224, 224))
    resized_image = tf.cast(resized_image, tf.uint8)

    example['image'] = resized_image
    return example


ds_train = ds_train.map(preprocess_image)
ds_val = ds_val.map(preprocess_image)
ds_test = ds_test.map(preprocess_image)


#Classification Labels

smoke_label_map = {0: "no", 1: "yes"}
density_label_map = {0: "none", 1: "low", 2: "medium", 3: "high"}

def get_label_counts(dataset, label_name, label_map=None):
    counts = Counter()
    for example in dataset:
        label = example[label_name].numpy()

        if label_map is not None:
            label = label_map.get(int(label), f"Unknown ({label})")
        counts[label] += 1
    return counts


# Smoke label distribution
smoke_counts = get_label_counts(ds_train, "smoke", smoke_label_map)
plt.bar(smoke_counts.keys(), smoke_counts.values(), color='orange')
plt.title("Smoke Label Distribution (Train)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# Density label distribution
density_counts = get_label_counts(ds_train, "density", density_label_map)
plt.bar(density_counts.keys(), density_counts.values(), color='skyblue')
plt.title("Density Label Distribution (Train)")
plt.xlabel("Density Level")
plt.ylabel("Count")
plt.show()



# Show first 3 images from training split
plt.figure(figsize=(10, 4))
for i, example in enumerate(ds_train.take(3)):
    image = example['image'].numpy()
    
    # Convert label indices to strings using dataset info
    smoke_label = info.features['smoke'].int2str(example['smoke'].numpy())
    density_label = info.features['density'].int2str(example['density'].numpy())

    plt.subplot(1, 3, i + 1)
    plt.imshow(image)
    plt.title(f"Smoke: {smoke_label}\nDensity: {density_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
