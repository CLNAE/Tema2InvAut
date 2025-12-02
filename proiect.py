import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# === Montare Google Drive și încărcare dataset ===
drive.mount('/content/drive')
stroke_set = pd.read_csv('/content/drive/MyDrive/Datasets/full_data.csv')

# === Preprocesare dataset ===
stroke_encoded = stroke_set.copy()
stroke_encoded['gender'] = stroke_encoded['gender'].map({'Male':0, 'Female':1})
stroke_encoded['ever_married'] = stroke_encoded['ever_married'].map({'No':0, 'Yes':1})
stroke_encoded['Residence_type'] = stroke_encoded['Residence_type'].map({'Urban':0, 'Rural':1})
stroke_numeric = stroke_encoded[['age','avg_glucose_level','bmi','hypertension',
                                 'heart_disease','stroke','gender','ever_married','Residence_type']]

# === Împărțire în caracteristici și țintă ===
X = stroke_numeric.drop('stroke', axis=1)
y = stroke_numeric['stroke']

# === Normalizare ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=63, stratify=y)

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# === Definirea modelelor ===
ridge_model = RidgeClassifier(alpha=1.0, random_state=42)
lasso_model = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42)

# === Dicționare pentru salvarea performanței ===
results = {'Model': [], 'Accuracy': [], 'Time': []}

# === Ridge Regression ===
start_time = time.time()
ridge_model.fit(X_train, y_train)
ridge_time = time.time() - start_time
y_pred_ridge = ridge_model.predict(X_test)
ridge_acc = accuracy_score(y_test, y_pred_ridge)

results['Model'].append('Ridge')
results['Accuracy'].append(ridge_acc)
results['Time'].append(ridge_time)

# === Lasso (Logistic Regression cu L1) ===
start_time = time.time()
lasso_model.fit(X_train, y_train)
lasso_time = time.time() - start_time
y_pred_lasso = lasso_model.predict(X_test)
lasso_acc = accuracy_score(y_test, y_pred_lasso)

results['Model'].append('Lasso')
results['Accuracy'].append(lasso_acc)
results['Time'].append(lasso_time)

# === Conversie rezultate în DataFrame ===
results_df = pd.DataFrame(results)
print(results_df)

# === Grafice de comparare ===
plt.figure(figsize=(12,5))

# Acuratețe
plt.subplot(1,2,1)
ax1 = sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Compararea acurateței')
plt.ylim(0,1)

# Adăugare valori numerice deasupra bar-urilor
for p in ax1.patches:
    height = p.get_height()
    ax1.text(p.get_x() + p.get_width()/2., height + 0.005, f'{height:.3f}', ha='center', va='bottom')

# Timp de execuție
plt.subplot(1,2,2)
ax2 = sns.barplot(x='Model', y='Time', data=results_df, palette='magma')
plt.title('Compararea timpului de execuție (secunde)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
corr = stroke_numeric.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matricea de corelație a variabilelor")
plt.show()
