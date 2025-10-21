from tensorflow.keras.models import load_model
import numpy as np

# Load your saved model
try:
    model = load_model('lung_cancer_cnn_model.keras')
    print("✅ Model loaded successfully.")

    # Dummy or test data to check if model gives reasonable output
    # Replace this with real test data
    test_input = np.random.rand(1, 224, 224, 3)  # For example, image input
    prediction = model.predict(test_input)

    print("🔍 Prediction from model:", prediction)

    # Check if output is not random or empty
    if prediction is not None and prediction.shape[0] > 0:
        print("✅ Model seems to be trained.")
    else:
        print("❌ Model is not trained or not working properly.")

except Exception as e:
    print("❌ Error loading model or predicting:", str(e))