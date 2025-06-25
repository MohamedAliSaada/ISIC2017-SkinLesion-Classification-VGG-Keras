import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Augmentation setup
data_gen =  ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.9, 1.1),
    horizontal_flip=True,
    fill_mode='nearest'
)


class FlexibleHFImageGenerator(Sequence):
    def __init__(self, hf_dataset, image_key="img", label_key="label", 
                 batch_size=32, shuffle=True, mode="none", target_size=None):
        """
        hf_dataset: Hugging Face dataset split
        image_key: name of image field
        label_key: name of label field
        batch_size: samples per batch (original count)
        shuffle: shuffle dataset after each epoch
        mode: 'none', 'augment', or 'double'
            - 'none': no augmentation
            - 'augment': augment and pass batch
            - 'double': original + augmented (batch_size * 2)
        target_size: (H, W) for resizing images
        """
        assert mode in ["none", "augment", "double"], "mode must be 'none', 'augment', or 'double'"
        
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.target_size = target_size
        self.indexes = np.arange(len(self.dataset))
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch"""
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        labels = []

        for i in batch_indexes:
            example = self.dataset[int(i)]
            image = example[self.image_key]

            # Ensure PIL image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Unsupported image format")

            # Resize if needed
            # you can make any extra enhancement here also not resize only 
            if self.target_size:
                image = image.resize(self.target_size)
            
            #Standard scaling
            image_np = np.array(image).astype(np.float32) / 255.0  # Standard scaling

            image_np = np.array(image)
            images.append(image_np)
            labels.append(example[self.label_key])

        images = np.array(images)
        labels = np.array(labels)

        if self.mode == "none":
            return images, labels
        
        elif self.mode == "augment":
            images_aug = next(data_gen.flow(images, batch_size=self.batch_size, shuffle=False))
            return images_aug, labels

        elif self.mode == "double":
            images_aug = next(data_gen.flow(images, batch_size=self.batch_size, shuffle=False))
            images_combined = np.concatenate([images, images_aug], axis=0)
            labels_combined = np.concatenate([labels, labels], axis=0)
            return images_combined, labels_combined

    def on_epoch_end(self):
        """Shuffle at epoch end"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
