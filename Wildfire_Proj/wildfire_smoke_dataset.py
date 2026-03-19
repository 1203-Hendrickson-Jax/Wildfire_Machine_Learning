import os
import csv
import tensorflow as tf
import tensorflow_datasets as tfds

class WildfireSmoke(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Wildfire smoke dataset labeled by presence and density. 2:1 imbalance with data (1290 smoke vs 648 non smoke)",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(224, 224, 3)),  # Images will be cropped during generation
                "smoke": tfds.features.ClassLabel(names=["no", "yes"]),
                "density": tfds.features.ClassLabel(names=["none", "low", "medium", "high"]),
                "file_name": tfds.features.Text(),  # For traceability
            }),
            supervised_keys=("image", "smoke"),
        )

    def _split_generators(self, dl_manager):
        data_dir = "D:/Wildfire_VD/labeled_output"
        csv_path = os.path.join(data_dir, "frame_labels.csv")

        with open(csv_path, newline='') as csvfile:
            reader = list(csv.DictReader(csvfile))
            random.seed(42)
            random.shuffle(reader)

        total = len(reader)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)

        train_rows = reader[:train_end]
        val_rows = reader[train_end:val_end]
        test_rows = reader[val_end:]

        return {
            "train": self._generate_examples(train_rows, data_dir),
            "validation": self._generate_examples(val_rows, data_dir),
            "test": self._generate_examples(test_rows, data_dir),
        }


    def _generate_examples(self, data_dir):
        csv_path = os.path.join(data_dir, "frame_labels.csv")
        frames_dir = os.path.join(data_dir, "frames")

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                filename = row["filename"]
                smoke = row["smoke"].strip().lower()
                density = row["density"].strip().lower()

                image_path = os.path.join(frames_dir, filename)
                if not os.path.exists(image_path):
                    continue  # Skip if file missing

                yield i, {
                    "image": self._load_and_crop(image_path),
                    "smoke": smoke,
                    "density": density,
                    "file_name": filename,
                }

    def _load_and_crop(self, image_path):
        def decode_and_crop(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            height = tf.shape(img)[0]
            cropped_img = img[:height - 30, :, :]  # Crop bottom 30 pixels
            resized_img = tf.image.resize(cropped_img, [224, 224])
            return resized_img

        return tfds.core.lazy_imports.tf_lazy(lambda: decode_and_crop(image_path))
