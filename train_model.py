import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load your dataset
data = pd.read_csv("weather_forecast_data.csv")  # Replace with your dataset filename

# Ensure the dataset has a "target" column
# Replace 'Rain' with your actual target column name if different
X = data.drop(columns=['Rain'])
y = data['Rain']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  # Only applicable for 'rbf' kernel
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='rain')  # Specify the positive label
recall = recall_score(y_test, y_pred, pos_label='rain')        # Specify the positive label

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model Precision: {precision:.2f}")
print(f"Model Recall: {recall:.2f}")

# Save the best model and scaler
joblib.dump(best_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Optimized model and scaler saved successfully!")
