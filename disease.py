from google.colab import files
files.upload()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("heart.csv")
# print first 5 rows of the dataset
df.head()
# print last 5 rows of the dataset
df.tail()
# number of rows and columns in the dataset
df.shape
# getting some info about the data
df.info()
# checking for missing values
df.isnull().sum()
# statistical measures about the data
df.describe()
df['target'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
X = df.drop('target', axis=1)
y = df['target']
print(X)
print(y)
print(X.shape)
print(y.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_acc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_acc)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)

print("SVC Accuracy:", svc_acc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(max_features=8, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_acc)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, svc_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVC")
plt.show()
# Dataset-based prediction check
y_pred = svc.predict(X_test)

print("Predicted values (first 10):", y_pred[:10])
print("Actual values (first 10):", y_test.values[:10])
import numpy as np

def predict_single_patient(patient_data):
    patient_data = np.array([patient_data])
    patient_data_scaled = scaler.transform(patient_data)
    prediction = svc.predict(patient_data_scaled)

    if prediction[0] == 1:
        return "Heart Disease Detected"
    else:
        return "No Heart Disease"
# Patient Data Format:
# [age (Age in years),
#  sex (Sex: 1 = Male, 0 = Female),
#  cp (Chest Pain Type),
#  trestbps (Resting Blood Pressure in mm Hg),
#  chol (Serum Cholesterol in mg/dl),
#  fbs (Fasting Blood Sugar > 120 mg/dl: 1 = True, 0 = False),
#  restecg (Resting Electrocardiographic Results),
#  thalach (Maximum Heart Rate Achieved),
#  exang (Exercise Induced Angina: 1 = Yes, 0 = No),
#  oldpeak (ST Depression induced by exercise),
#  slope (Slope of the peak exercise ST segment),
#  ca (Number of major vessels colored by fluoroscopy),
#  thal (Thalassemia: 1 = Normal, 2 = Fixed defect, 3 = Reversible defect)
# ]

import pandas as pd

# Two patient records together
patient_input_df = pd.DataFrame([
    {
        # Patient 1: Heart Disease Data
        'age': 62,
        'sex': 0,
        'cp': 0,
        'trestbps': 140,
        'chol': 268,
        'fbs': 0,
        'restecg': 0,
        'thalach': 160,
        'exang': 0,
        'oldpeak': 3.6,
        'slope': 0,
        'ca': 2,
        'thal': 2
    },
    {
        # Patient 2: Heart Disease Data
        'age': 52,
        'sex': 1,
        'cp': 0,
        'trestbps': 128,
        'chol': 204,
        'fbs': 1,
        'restecg': 1,
        'thalach': 156,
        'exang': 1,
        'oldpeak': 1,
        'slope': 1,
        'ca': 0,
        'thal': 0
    }
])

# Scale and predict
patient_scaled = scaler.transform(patient_input_df)
predictions = svc.predict(patient_scaled)

# Show results for each patient
for i, pred in enumerate(predictions, start=1):
    if pred == 1:
        print(f"Patient {i}: Heart Disease Detected")
    else:
        print(f"Patient {i}: No Heart Disease")

print("\nPrediction Probability:")

proba = svc.predict_proba(patient_scaled)

for i, p in enumerate(proba, start=1):
    print(f"Patient {i} → No Disease: {p[0]:.2f}, Disease: {p[1]:.2f}")