import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("data.csv")

# Drop unnecessary columns
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# Encode the diagnosis column (Malignant=1, Benign=0)
data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])

# Select only 5 features for simplicity
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
X = data[features]
y = data['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'breast_cancer_model.pkl')
print("Model trained and saved successfully.")
