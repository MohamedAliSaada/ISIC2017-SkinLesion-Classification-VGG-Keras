from tensorflow.keras.callbacks import EarlyStopping , Callback
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import sys

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
        sys.stdout.flush()  # مهم عشان نضمن الترتيب
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
