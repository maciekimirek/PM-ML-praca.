import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('../../Data_source/Event_log.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'],format = "mixed")

#agregacja danych potrzebna do pokazania parametrow per order
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

#definiowane X i Y do modelu
y = agg['total_duration_hours']
X = agg[['order_value', 'items', 'discount', 'region', 'source']]

#Mamy wartosci numeryczne, ktore przechodza dalej ale teg kategoryczne, wiec trzeba je
#odeparowac i odpowiednio zakodowac
#Kolumny ktore pobieramy
num_features = ['order_value', 'items', 'discount']
cat_features = ['region', 'source']

#Podstawiamy do X
X_num = X[num_features]
X_cat = X[cat_features]

#One-hot dla X_cat
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_cat)

#Nazwy wygenerowanych kolumn kategorii
cat_feature_names = encoder.get_feature_names_out(cat_features)

#Skalowanie
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

#Złożenie końcowej macierzy
X_final = np.hstack([X_num_scaled, X_cat_encoded])

#Train-test
X_train, X_test, y_train, y_test = train_test_split( X_final, y, test_size=0.2, random_state=42
)

#Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Metryki
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=== Metryki (test) ===")
print(f"MAE : {mae:.3f} h")
print(f"RMSE: {rmse:.3f} h")
print(f"R²  : {r2:.3f}")


#Wniosek: regresja liniowa nie jest najlepszym wyborem, poniewaz dane w event logu nie sa liniowe.
#Komentarz i wnioski do rozbudowania

