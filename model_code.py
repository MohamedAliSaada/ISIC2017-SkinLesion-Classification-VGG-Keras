# -*- coding: utf-8 -*-
"""model_code

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TsLUJTSYYZWWJfpN9ii5IyMkJBuwXFJG

# start
"""

!pip install -U tensorflow datasets

from collections import Counter
from tqdm import tqdm

#load data sets from hugging face

from datasets import load_dataset

ds = load_dataset("TynClause/isic-2017-dataset")

#get the train part and validation part
ds_train = ds['train'].shuffle(seed=100)          #{0: 374, 1: 1372, 2: 254}
ds_validation =ds['validation'].shuffle(seed=100) #{0: 30, 1: 78, 2: 42}

"""# class to make data on fly"""

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
        super().__init__()  # Important for compatibility
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

    def __getitem__(self, idx):  #this pass by fit to get batchs 0,1,2,3 and so on
        """Generate one batch"""

        #batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        #or safly use :
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.indexes)) #so if he get 0-128 but i have only 100 , he handel this correctly .
        batch_indexes = self.indexes[start:end]

        if len(batch_indexes) == 0:
          raise IndexError(f"Empty batch generated at idx={idx}")

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


            images.append(image_np)
            labels.append(example[self.label_key])

        images = np.array(images)
        labels = np.array(labels)

        if self.mode == "none":
            return images, labels

        elif self.mode == "augment":
            images_aug = next(data_gen.flow(images, batch_size=images.shape[0], shuffle=False))
            return images_aug, labels

        elif self.mode == "double":
            images_aug = next(data_gen.flow(images, batch_size=images.shape[0], shuffle=False))
            images_combined = np.concatenate([images, images_aug], axis=0)
            labels_combined = np.concatenate([labels, labels], axis=0)
            return images_combined, labels_combined

    def on_epoch_end(self):
        """Shuffle at epoch end"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

"""# define custom call back"""

from tensorflow.keras.callbacks import EarlyStopping , Callback ,ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np




reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=0, min_lr=1e-7 ,verbose=1)

early_stopping =  EarlyStopping(
    monitor="val_loss",
    patience=3,
    verbose=1,
    restore_best_weights=True
    )

#any dictinary has it's own keys (  data.keys()  ) make it list [   list(data.keys())  ]
#name = "Ali"
#print("Hello, {}".format(name)) this give "Hello, Ali" as it's like formated string

class custom_callback(Callback):
  def __init__(self , train_gen=None ,val_gen=None):
    super().__init__()
    self.train_gen=train_gen
    self.val_gen=val_gen

  def compute_metrics(self ,generator ):
    y_pred=[]
    y_true=[]
    for batch_x , batch_y in generator:
      preds = self.model.predict(batch_x , verbose=0)
      preds_classes = np.argmax(preds, axis=1)  #this is now y_pred

      y_true.extend(batch_y)
      y_pred.extend(preds_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return precision, recall, f1

  def on_epoch_end(self, epoch, logs=None):

        print(f"\n--- Epoch {epoch+1} Metrics ---")

        # Training metrics
        if self.train_gen is not None:
            precision, recall, f1 = self.compute_metrics(self.train_gen)
            print(f"Training -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Validation metrics
        if self.val_gen is not None:
            precision, recall, f1 = self.compute_metrics(self.val_gen)
            print(f"Validation -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        print("------------------------------\n")

"""# end"""

#make my data ready
ds_t = FlexibleHFImageGenerator(ds_train, batch_size=16, mode="double", target_size=(224, 224),image_key="image", label_key="label")
ds_v = FlexibleHFImageGenerator(ds_validation, batch_size=16, mode="none", target_size=(224, 224),image_key="image", label_key="label")
print(f"Batches per epoch: {len(ds_t)}")
print(f"Batches per epoch: {len(ds_v)}")

my_callback = custom_callback(ds_t,ds_v)

#now let's build the model
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten , Dense , Dropout
from tensorflow.keras.models import Model

# Load VGG16 base
base_model =  VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

for  i in range(len(base_model.layers)):
  if i >= 15:
    base_model.layers[i].trainable=True
  else :
    base_model.layers[i].trainable=False

# add custom layer to vgg16 base model
x = Flatten()(base_model.output)
x = Dense(256,activation='relu',name='d1')(x)
x = Dropout(.5)(x)
x = Dense(3,activation='softmax')(x)

#combine now the two
model = Model(base_model.input,x)
model.summary()

#make model compile setting
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#make model fit setting
model.fit(
    ds_t,
    epochs=15,
    validation_data=ds_v,
    callbacks=[early_stopping,my_callback,reduce_lr]

)

