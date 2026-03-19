import os
import csv
import random
import tensorflow as tf
import tensorflow_datasets as tfds

class WildfireSmoke(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Wildfire smoke dataset labeled by presence and density.",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(224, 224, 3)),
                "smoke": tfds.features.ClassLabel(names=["no", "yes"]),
                "density": tfds.features.ClassLabel(names=["none", "low", "medium", "high"]),
                "file_name": tfds.features.Text(),
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

    def _generate_examples(self, rows, data_dir):
        frames_dir = os.path.join(data_dir, "frames")

        for i, row in enumerate(rows):
            filename = row["filename"]
            smoke = row["smoke"].strip().lower()
            density = row["density"].strip().lower()
            image_path = os.path.join(frames_dir, filename)

            if not os.path.exists(image_path):
                continue

            yield i, {
                "image": image_path,
                "smoke": smoke,
                "density": density,
                "file_name": filename,
            }
