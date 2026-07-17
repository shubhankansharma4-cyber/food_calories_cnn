from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_model

# -----------------------------
# Image Parameters
# -----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"

# -----------------------------
# Data Preprocessing
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1./255
)

# -----------------------------
# Load Training Data
# -----------------------------
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -----------------------------
# Load Validation Data
# -----------------------------
validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -----------------------------
# Build CNN Model
# -----------------------------
model = build_model()

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "food_classifier.keras",
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# -----------------------------
# Save Final Model
# -----------------------------
model.save("final_food_classifier.keras")
