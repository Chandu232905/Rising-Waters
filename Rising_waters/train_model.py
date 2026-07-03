import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

# ---------------------------------------
# Load Dataset
# ---------------------------------------
data = pd.read_csv("dataset/flood_prediction_dataset.csv")

print("Dataset Loaded Successfully!\n")

# ---------------------------------------
# Select Features and Target
# ---------------------------------------
X = data[[
    "Annual_Rainfall",
    "Monsoon_Rainfall",
    "Cloud_Visibility_km",
    "Humidity_pct",
    "River_Level_Index"
]]

y = data["FloodOccurred"]

# ---------------------------------------
# Split Dataset
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# Create Models
# ---------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )
}

best_model = None
best_model_name = ""
best_accuracy = 0

print("=" * 50)
print("MODEL ACCURACY")
print("=" * 50)

# ---------------------------------------
# Train All Models
# ---------------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name} : {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ---------------------------------------
# Save Best Model
# ---------------------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/flood_model.pkl")

print("\n" + "=" * 50)
print("BEST MODEL")
print("=" * 50)
print("Model :", best_model_name)
print(f"Accuracy : {best_accuracy * 100:.2f}%")

print("\nModel saved as:")
print("model/flood_model.pkl")