import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# Load dataset
data = pd.read_csv("../data/dataR2.csv")

# Adjust target labels: 1 (Healthy) -> 1, 2 (Cancer) -> 0
data['Classification'] = data['Classification'].map({1: 1, 2: 0})
X = data.drop('Classification', axis=1).values
y = data['Classification'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define XGBoost model
base_model = XGBClassifier(scale_pos_weight=2.0, random_state=42)  # Increased weight for Healthy class

# Define hyperparameter grid for tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Perform grid search
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_
from xgboost import plot_importance
plt.figure(figsize=(8, 6))
plot_importance(best_model)
plt.title("Feature Importance")
plt.savefig("../output/feature_importance.jpeg", dpi=300, bbox_inches="tight")
plt.close()

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train_scaled, y_train)

# Predict probabilities
probas = calibrated_model.predict_proba(X_test_scaled)

# Adjust threshold
threshold = 0.5
y_pred = (probas[:, 1] >= threshold).astype(int)  # Updated logic: Use Healthy probability

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', pos_label=0)  # For Cancer (0)
recall = recall_score(y_test, y_pred, average='binary', pos_label=0)
f1 = f1_score(y_test, y_pred, average='binary', pos_label=0)

# Save metrics to output.txt
with open("../output/output.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision (Cancer): {precision}\n")
    f.write(f"Recall (Cancer): {recall}\n")
    f.write(f"F1-Score (Cancer): {f1}\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best Score (Balanced Accuracy): {grid_search.best_score_}\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
            xticklabels=["Cancer (0)", "Healthy (1)"],
            yticklabels=["Cancer (0)", "Healthy (1)"])
plt.title("Confusion Matrix", fontsize=14, pad=15)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.savefig("../output/prediction_counts.jpeg", dpi=300, bbox_inches="tight")
plt.close()

# Save the model and scaler
joblib.dump(calibrated_model, "../output/breast_cancer_model.pkl")
joblib.dump(scaler, "../output/scaler.pkl")