import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv('dup.csv')

# Define feature and target columns
selected_features = ['Dissolved Oxygen (mg/L)', 'Salinity (ppt)', 'pH', 'Water Temp', 'Ammonia (mg/L)', 'Season']
target_column = 'Disease to Occur'  
# Ensure all numeric columns are converted correctly
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Convert numeric columns properly
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')  
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Fill numeric NaNs with mean

# Fill missing values in categorical columns with mode
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Convert target column to multi-label format
data[target_column] = data[target_column].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Encode target using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data[target_column])
data = data.drop(columns=[target_column])
# Encode 'Season' using One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
season_encoded = encoder.fit_transform(data[['Season']])
season_encoded_df = pd.DataFrame(season_encoded, columns=encoder.get_feature_names_out(['Season']))

# Drop original 'Season' column and add encoded values
data = data.drop(columns=['Season'])
data = pd.concat([data, season_encoded_df], axis=1)

# Select features
X = data[selected_features[:-1]]  # Exclude 'Season' since it's now encoded separately
X = pd.concat([X, season_encoded_df], axis=1)  # Add encoded Season data

# Debugging: Check for non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print("❌ Non-numeric columns found in X:", non_numeric_cols)
    print(X[non_numeric_cols].head())  # Print first few rows for debugging
    raise ValueError("Non-numeric data found in X. Check preprocessing.")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA 
pca = PCA(n_components=min(X_scaled.shape[1], 6))
X_pca = pca.fit_transform(X_scaled)

# Handle class imbalance using SMOTE (applied per label)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Multi-Output RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,random_state=42)
multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, hamming_loss

# Predict on test data
y_pred = multi_rf.predict(X_test)

# Convert predictions back to original labels
y_pred_labels = mlb.inverse_transform(y_pred)
y_test_labels = mlb.inverse_transform(y_test)

# Calculate accuracy for each label
accuracy_per_label = np.mean(y_pred == y_test, axis=0)
overall_accuracy = np.mean(accuracy_per_label)  # Mean accuracy across all labels

# Print classification report
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
# Check unique predictions
unique_preds = np.unique(y_pred, axis=0)
print(f"\n🛠 Unique Predictions in Test Data: {unique_preds.shape[0]}")

# Check distribution of predicted labels
print(f"\n📊 Prediction Distribution: {np.sum(y_pred, axis=0)}")
print(f"\n📊 True Label Distribution: {np.sum(y_test, axis=0)}")

# Print overall accuracy
print(f"\n✅ Overall Model Accuracy: {overall_accuracy * 100:.2f}%")

# Save models
with open('fish_disease_model.pkl', 'wb') as f:
    pickle.dump(multi_rf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
# Save feature order (important for prediction API)
feature_order = selected_features[:-1]  # Exclude 'Season' (added separately)
with open('feature_order.pkl', 'wb') as f:
    pickle.dump(feature_order, f)

# Save column names for debugging
with open('column_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and preprocessing steps saved successfully!")

print("✅ Model training complete. All models saved successfully!")