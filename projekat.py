import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from statsmodels.api import add_constant
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from statsmodels.stats.stattools import durbin_watson
import joblib

def linearity_assumption(model, features, labels):
    predicted_values = model.predict(features)
    residuals = labels - predicted_values
    linearity_satisfied = np.allclose(residuals, 0, atol=1e-5)
    return linearity_satisfied

#ucitavanje podataka
data = pd.read_csv('DelayedFlights.csv')

#izbacivanje nebitnih kolona
irrelevant_columns = ['Unnamed: 0', 'TailNum', 'CancellationCode', 'Diverted',
                      'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'Year', 'LateAircraftDelay',
                      'TaxiIn']

data_cleaned = data.drop(columns=irrelevant_columns)

#provera null vrednosti
print("Nedostajuće vrednosti:\n", data_cleaned.isnull().sum())


data_sample = data_cleaned.sample(n=10000, random_state=42)

print(data_sample.dtypes)
data_sample = data_sample.select_dtypes(include=[np.number])
#provera da li su sve kolone sada numericke
print(data_sample.dtypes)

#popunjavanje nedostajucih vrenodsti srednjom vrednosti
data_filled = data_sample.fillna(data_sample.mean())

#izbacivanje visoko koreliranih kolona
corr_matrix = data_filled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
data_reduced = data_filled.drop(columns=to_drop)

#matrica korelacije
correlation_matrix = data_reduced.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
plt.title('Matrica korelacije (nakon izbacivanja visoko koreliranih kolona)')
plt.show()

#analiza distribucije podataka
data_reduced.hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribucija podataka')
plt.show()

#provera outliera
plt.figure(figsize=(15, 10))
sns.boxplot(data=data_reduced)
plt.title('Boxplot podataka za identifikaciju ekstremnih vrednosti')
plt.show()

def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_out

data_no_outliers = remove_outliers_iqr(data_reduced)

#nnova distribucija podataka
data_no_outliers.hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribucija podataka (nakon uklanjanja outliera)')
plt.show()

#outlieri nakon izbacivanja
plt.figure(figsize=(15, 10))
sns.boxplot(data=data_no_outliers)
plt.title('Boxplot podataka za identifikaciju ekstremnih vrednosti (nakon uklanjanja outliera)')
plt.show()

print(data_no_outliers["Cancelled"])

cancelled_counts = data_no_outliers["Cancelled"].value_counts()
print("Broj 0 u koloni 'Cancelled':", cancelled_counts[0])
print("Broj 1 u koloni 'Cancelled':", cancelled_counts[1])

#balansiranje podataka - SMOTE
if data_no_outliers['Cancelled'].nunique() == 1:
    data_no_outliers = data_reduced.copy()
else:
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(data_no_outliers.drop(columns=['Cancelled']),
                                                data_no_outliers['Cancelled'])
    data_no_outliers = pd.concat([X_balanced, pd.DataFrame(y_balanced, columns=['Cancelled'])], axis=1)


#####

target = 'Cancelled'
features = data_no_outliers.drop(columns=[target]).columns

#podela podataka na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(data_no_outliers[features], data_no_outliers[target], test_size=0.2,
                                                    random_state=42)


logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

#predikcija i evaluacija logisticke reg
y_pred_logistic = logistic_model.predict(X_test)
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_logistic))
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_logistic))
print('Logistic Regression Confusion Matrix :\n', confusion_matrix(y_test, y_pred_logistic))

#provera linearne zavisnosti 
linearity_satisfied = linearity_assumption(logistic_model, X_train, y_train)
print("Linearna zavisnost zadovoljena:", linearity_satisfied)



## random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report (pre scalinga)")
print(classification_report(y_test, y_pred_rf))
print('Random Forest Accuracy (pre scalinga):', accuracy_score(y_test, y_pred_rf))
print('Random Forest Confusion Matrix (pre scalinga):\n', confusion_matrix(y_test, y_pred_rf))


#normalizacija podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#rf nad normalizovanim
rf_model_scaled = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_scaled.fit(X_train_scaled, y_train)


y_pred_rf_scaled = rf_model_scaled.predict(X_test_scaled)
print("Random Forest Classification Report (nakon scalinga)")
print(classification_report(y_test, y_pred_rf_scaled))
print('Random Forest Accuracy (nakon scalinga):', accuracy_score(y_test, y_pred_rf_scaled))
print('Random Forest Confusion Matrix (nakon scalinga):\n', confusion_matrix(y_test, y_pred_rf_scaled))


#polinomijalna transformacija
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train) ## scaled
X_test_poly = poly.transform(X_test) ## scaled

#rf polinomijalni
rf_model_poly = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_poly.fit(X_train_poly, y_train)


y_pred_rf_poly = rf_model_poly.predict(X_test_poly)
print("Random Forest Classification Report (nakon polinomijalne transformacije)")
print(classification_report(y_test, y_pred_rf_poly))
print('Random Forest Accuracy (nakon polinomijalne transformacije):', accuracy_score(y_test, y_pred_rf_poly))
print('Random Forest Confusion Matrix (nakon polinomijalne transformacije):\n',
      confusion_matrix(y_test, y_pred_rf_poly))

#randomized search za optimizaciju hiperparametara
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_dist, n_iter=200, cv=5, n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_

# rf sa najboljim hiperparametrima
rf_best_model = RandomForestClassifier(**best_params, random_state=42)
rf_best_model.fit(X_train, y_train)

y_pred_rf_best = rf_best_model.predict(X_test)
print("Random Forest Classification Report (sa najboljim hiperparametrima)")
print(classification_report(y_test, y_pred_rf_best))
print('Random Forest Accuracy (sa najboljim hiperparametrima):', accuracy_score(y_test, y_pred_rf_best))
print('Random Forest Confusion Matrix (sa najboljim hiperparametrima):\n', confusion_matrix(y_test, y_pred_rf_best))

#cuvanje modela
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(rf_model_scaled, 'rf_model_scaled.pkl')
joblib.dump(rf_model_poly, 'rf_model_poly.pkl')
joblib.dump(rf_best_model, 'rf_best_model.pkl')
