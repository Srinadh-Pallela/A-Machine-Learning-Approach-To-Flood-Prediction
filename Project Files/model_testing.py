import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Used to save the model for the web app
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import the models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# --- STEP 1: LOAD AND PREPROCESS ---
df = pd.read_csv('flood_risk.csv')

# FIX: Separate LabelEncoders to prevent mapping collision
# This ensures Land Cover categories don't overwrite Soil Type categories
le_land = LabelEncoder()
df['Land Cover'] = le_land.fit_transform(df['Land Cover'])

le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

# Feature Selection (Independent Variables X, Dependent Variable y)
X = df.drop('Flood Occurred', axis=1)
y = df['Flood Occurred']

# Scaling the independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- STEP 2: DEFINE AND TEST MODELS ---
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}

print("--- Accuracy Comparison ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    results[name] = acc
    print(f"{name}: {acc*100:.2f}%")

# --- STEP 3: SAVE ASSETS FOR WEB APPLICATION ---
# We use Random Forest as the production model
best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test)

# Save all assets so the Web App can use them
joblib.dump(best_model, 'flood_model.pkl')
joblib.dump(scaler, 'flood_scaler.pkl')
joblib.dump(le_land, 'land_encoder.pkl')
joblib.dump(le_soil, 'soil_encoder.pkl')

print("\n✅ SUCCESS: Model, Scaler, and Encoders saved as .pkl files!")

# --- STEP 4: DETAILED EVALUATION ---
print("\n--- Detailed Report (Random Forest) ---")
print(classification_report(y_test, y_pred_best))

# --- STEP 5: PREPARE VISUALIZATIONS ---

# 1. Correlation Heatmap

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('1. Correlation Heatmap of Features')
plt.tight_layout()

# 2. Model Accuracy Comparison (Fixed warning line)
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracy_values = list(results.values())
sns.barplot(x=model_names, y=accuracy_values, hue=model_names, palette='viridis', legend=False)
plt.title('2. Comparison of Model Accuracies')
plt.ylabel('Accuracy Score')
plt.ylim(0.7, 1.0) 
plt.tight_layout()

# 3. Feature Importance Plot

plt.figure(figsize=(10, 6))
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='skyblue')
plt.title('3. Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()

# 4. Confusion Matrix

plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('4. Confusion Matrix: Random Forest')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# --- STEP 6: SHOW ALL GRAPHS ---
print("\nOpening all visualization windows simultaneously...")
plt.show()