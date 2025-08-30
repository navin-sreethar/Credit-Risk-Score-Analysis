import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
DATA_PATH = os.path.join('data', 'credit_data.csv')
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_model.pkl')

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Print column names to see what we're working with
print("Dataset columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Find the target column (default payment column)
default_column = None
possible_default_columns = ['default', 'default.payment.next.month', 'DEFAULT', 'Default', 
                          'defaultPaymentNextMonth', 'default payment next month']
for col in possible_default_columns:
    if col in df.columns:
        default_column = col
        print(f"Found target column: '{default_column}'")
        if col != 'default':
            df = df.rename(columns={col: 'default'})
            print(f"Renamed '{col}' to 'default'")
        break

if default_column is None:
    print("Could not find the default payment column. Available columns are:")
    print(df.columns.tolist())
    raise ValueError("Target column not found. Please check your CSV file.")

# Drop ID column if present
id_columns = ['ID', 'Id', 'id']
for col in id_columns:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped column: '{col}'")

# Features and target
print("Target column:", 'default')
X = df.drop('default', axis=1)
y = df['default']

# Print class distribution
print(f"Class distribution:\n{y.value_counts()}")
print(f"Default rate: {y.mean():.2%}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Scale the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature engineering
print("\nPerforming feature engineering...")

# 1. Calculate debt ratio (total bill amount / limit balance)
X_train['DEBT_RATIO'] = X_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1) / X_train['LIMIT_BAL'].replace(0, 1)
X_test['DEBT_RATIO'] = X_test[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1) / X_test['LIMIT_BAL'].replace(0, 1)

# 2. Calculate payment ratio (payment / bill)
for i in range(1, 7):
    bill_col = f'BILL_AMT{i}'
    pay_col = f'PAY_AMT{i}'
    ratio_col = f'PAY_RATIO_{i}'
    
    X_train[ratio_col] = X_train[pay_col] / X_train[bill_col].replace(0, 1)
    X_train[ratio_col] = X_train[ratio_col].replace([np.inf, -np.inf], 0).fillna(0)
    
    X_test[ratio_col] = X_test[pay_col] / X_test[bill_col].replace(0, 1)
    X_test[ratio_col] = X_test[ratio_col].replace([np.inf, -np.inf], 0).fillna(0)

# 3. Calculate average payment delay
X_train['AVG_PAY_DELAY'] = X_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
X_test['AVG_PAY_DELAY'] = X_test[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)

# 4. Binary indicator of previous delay
X_train['HAS_PREV_DELAY'] = ((X_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1) > 0).astype(int)
X_test['HAS_PREV_DELAY'] = ((X_test[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1) > 0).astype(int)

# 5. Number of months with delay
X_train['NUM_DELAY_MONTHS'] = (X_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1)
X_test['NUM_DELAY_MONTHS'] = (X_test[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1)

print(f"Added new features. New shape: {X_train.shape}")

# Update scaled features after engineering
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

best_model = None
best_score = 0
best_model_name = None

print("\nModel Comparison:")
for name, model in models.items():
    # Perform cross-validation on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    print(f"{name} - Mean CV ROC-AUC: {mean_score:.4f} (±{std_score:.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with ROC-AUC: {best_score:.4f}")

# Make sure best_model is defined even if the comparison fails
if best_model is None:
    print("Defaulting to Random Forest as no best model was found")
    best_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    best_model_name = 'Random Forest'

# Fit the best model on the full training set
best_model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]  # Probability of default

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Print results
print("\nTest Set Results:")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Feature importance
try:
    if best_model_name == 'Logistic Regression':
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(best_model.coef_[0])
        })
    elif best_model_name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        })
    else:  # Fallback for any other model type
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.zeros(len(X_train.columns))
        })
    
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
except Exception as e:
    print(f"\nCould not display feature importance: {e}")

# Save model and scaler
joblib.dump(best_model, MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
print(f'Scaler saved to {os.path.join(MODEL_DIR, "scaler.pkl")}')
