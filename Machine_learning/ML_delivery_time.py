import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler




df = pd.read_csv('../Data_source/Event_log.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format = "mixed")
agg = df.groupby('Order_ID').agg(
    start_time=('Timestamp', 'min'),
    end_time=('Timestamp', 'max'),
    total_duration_hours=('Timestamp', lambda x: (x.max() - x.min()).total_seconds() / 3600.0),
    order_value=('Order_Value', 'first'),
    items=('Items_Count', 'first'),
    discount=('Discount', 'first'),
    region=('Region', 'first'),
    source=('Order_Source', 'first')
).reset_index()

def classify_duration(h):
    if h < 24: return "fast"
    if h < 48: return "medium"
    else: return "slow"

agg['duration_class'] = agg['total_duration_hours'].apply(classify_duration)

y = agg['duration_class']
X = agg[['order_value', 'items', 'discount', 'region', 'source']]

X_num = agg[['order_value', 'items', 'discount']]
X_cat = agg[['region', 'source']]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_cat)

X_final = np.hstack([X_cat_encoded, X_num])

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size = 0.2, random_state= 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
print("RandomForest — accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=rf.classes_, columns=rf.classes_))



# KNN
knn = KNeighborsClassifier(n_neighbors=5)   # możesz eksperymentować z n_neighbors
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("KNN — accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=knn.classes_, columns=knn.classes_))



# SVM
svc = SVC(kernel='rbf', probability=False, random_state=42)
svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test_scaled)
print("SVC — accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=svc.classes_, columns=svc.classes_))

print(agg['duration_class'].value_counts())

#Wynki nie sa zadawalajace, trzeba to rozwinac o wieksza ilosc parametrow.



