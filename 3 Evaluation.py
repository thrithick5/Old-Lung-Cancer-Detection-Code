from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.models import load_model
from tensorflow.preprocessing.image import ImageDataGenerator

# Load your saved model
model = load_model('lung_cancer_cnn_model.keras')

# Define the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_dir = 'processed_dataset/train' 
test_dir = 'processed_dataset/val' 
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Generate predictions
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=-1)
y_proba = model.predict(test_gen)[:, 1]  # Assuming binary classification, adjust if necessary

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_true, y_proba)
auc = roc_auc_score(y_true, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()