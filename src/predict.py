import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("food_classifier.keras")

# Class names (same order as training)
class_names = [
    "pizza",
    "burger",
    "pasta",
    "salad",
    "apple_pie",
    "fried_rice",
    "ice_cream",
    "sushi"
]

# Calories per 100g
calorie_dict = {
    "pizza": 285,
    "burger": 295,
    "pasta": 131,
    "salad": 33,
    "apple_pie": 237,
    "fried_rice": 163,
    "ice_cream": 207,
    "sushi": 130
}

# Load and preprocess image
img = image.load_img("test.jpg", target_size=(224, 224))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)

# Get predicted class
predicted_index = np.argmax(prediction)
predicted_food = class_names[predicted_index]

# Get calories
calories = calorie_dict[predicted_food]

# Final output
print(f"Estimated Calories: {calories} kcal per 100g")
