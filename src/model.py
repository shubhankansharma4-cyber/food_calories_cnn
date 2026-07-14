# src/model.py

# src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


def build_model(input_shape=(224, 224, 3), num_classes=25):

    model = Sequential()

    # ---------------- First Convolution Block ----------------
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ---------------- Second Convolution Block ----------------
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ---------------- Third Convolution Block ----------------
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu'
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ---------------- Classification Head ----------------
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    # Reduce overfitting
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile Model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
