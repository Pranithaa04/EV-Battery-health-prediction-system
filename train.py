import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_excel("D:/internship/ev_.xlsx")

# Feature engineering
df["avg_temperature"] = df["Temperature (Â°C)"]
df["daily_distance_km"] = df["Distance Driven (since last charge) (km)"]
df["fast_charging_ratio"] = df["Charging Rate (kW)"] / 100
df["battery_age_months"] = df["Vehicle Age (years)"] * 12

# Simulate additional features
np.random.seed(42)
df["charge_cycles"] = np.random.randint(200, 1600, size=len(df))
df["idle_days"] = np.random.randint(0, 15, size=len(df))

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)


# Generate synthetic labels
df["battery_health"] = 0  # Healthy
df.loc[((df["battery_age_months"] > 60) | (df["fast_charging_ratio"] > 0.6)) & (np.random.rand(len(df)) > 0.3), "battery_health"] = 1  # Degrading
df.loc[(df["charge_cycles"] > 1300) & (np.random.rand(len(df)) > 0.5), "battery_health"] = 2  # Critical

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = df[["avg_temperature","charge_cycles","daily_distance_km","fast_charging_ratio","idle_days","battery_age_months"]]
y = df["battery_health"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


model_accuracies = {}
# -------------------------------
# Define Multiple Models
# -------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=30000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier()
}

# -------------------------------
# Train and Evaluate All Models
# -------------------------------
for name, model in models.items():
    print(f"\n Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    model_accuracies[name] = acc

    print(f" Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if name == "Random Forest":
        # Save best-performing model
        with open("battery_model.pkl", "wb") as f:
            pickle.dump(model, f)

#plot the graph
plt.figure(figsize=(10, 6))
bars = plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['green', 'blue', 'purple', 'orange', 'red'])
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
