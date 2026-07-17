from tensorflow.keras.models import load_model
import numpy as np

# Load trained model
model = load_model("food_classifier.keras")

# Calorie lookup dictionary
calorie_dict = {
    "pizza": 285,
    "burger": 295,
    "pasta": 131,
    ...
}

# Load image
...

# Predict class
prediction = model.predict(img)

predicted_index = np.argmax(prediction)

predicted_food = class_names[predicted_index]

# Lookup calories
calories = calorie_dict[predicted_food]

print(predicted_food)
print(calories)
