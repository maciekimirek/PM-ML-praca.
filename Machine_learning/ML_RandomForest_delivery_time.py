import pandas as pd
import pm4py
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("Event_log.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format = "mixed")

#Wyodrebnianie cech procesowych
#Tworzemy funkcje, bo możemy mieć elementy, które nie maja danych aktywności. Wewnątrz funkcji korzystamy z pętli,
#aby troche uprościć kod. Dobrze byłoby zrobić "first_activity" i "second_acivity" dynamicznie. Do przemyślenia.

def compute_hours(g):
    first_activity = ["Złożenie zamówienia", "Weryfikacja", "Złożenie zamówienia", "Płatność",
                      "Wysyłka", "Dostawa", "Dostawa", "Dostawa", "Zwrot"]
    second_activity = ["Weryfikacja", "Płatność", "Płatność", "Wysyłka",
                       "Dostawa", "Ocena", "Zwrot", "Reklamacja", "Ocena"]

    results = {}

    for a1, a2 in zip(first_activity, second_activity):
        try:
            ts1 = g.loc[g["Activity"] == a1, "Timestamp"].iloc[0]
            ts2 = g.loc[g["Activity"] == a2, "Timestamp"].iloc[0]
            hours = (ts2 - ts1).total_seconds() / 3600
        except IndexError:
            hours = None  # brak danych

        colname = f"hours_{a1}_to_{a2}"
        results[colname] = hours

    return pd.Series(results)

#Pierwsza czesc dataframe
df_cycletime = df.groupby("Order_ID").apply(compute_hours)

#Wyodrębnienie wariantów, ilości aktywności i nadanie numeru konkretnemu wariantowi, aby moć te dane dopisać do oryginalnego DF i uzycz w Machine Learningu
#Przygotowanie df do pracy z pm4py
df = pm4py.format_dataframe(df, case_id='Order_ID', activity_key='Activity', timestamp_key='Timestamp')

rows = []
variant_counter = 1

for variant, subdf in pm4py.split_by_process_variant(df):
    variant_name = variant_counter

    for order_id in subdf["Order_ID"].unique():
        rows.append({
            "Order_ID": order_id,
            "variant_original": variant,
            "variant_name": variant_name
        })

    variant_counter += 1

#Druga czesc dataframe
#Stworzenie DF z wierszy, ktore powstaly w rows.append
df_variants = pd.DataFrame(rows)
#Utworzenie kolumny na podstawie ilosci elementow w kolumnie "variant_original"
df_variants["activities_count"] = df_variants["variant_original"].apply(len)

df_merged = df_cycletime.merge(df_variants, on="Order_ID", how="left")



###Odpalamy machine learning tak jak w ML_delivery_time ale z dodatkowymi cechami procesowymi

#Tworzenie DF z poczatkowymi cechami

df_agg = df.groupby('Order_ID').agg(
    start_time=('Timestamp', 'min'),
    end_time=('Timestamp', 'max'),
    total_duration_hours=('Timestamp', lambda x: (x.max() - x.min()).total_seconds() / 3600.0),
    order_value=('Order_Value', 'first'),
    items=('Items_Count', 'first'),
    discount=('Discount', 'first'),
    region=('Region', 'first'),
    source=('Order_Source', 'first')
).reset_index()


##Laczenie

df_final = df_merged.merge(df_agg, on="Order_ID", how="left")


def classify_duration(h):
    if h < 24: return "fast"
    if h < 48: return "medium"
    else: return "slow"

df_final['duration_class'] = df_final['total_duration_hours'].apply(classify_duration)

y = df_final['duration_class']
X = df_final[['hours_Złożenie zamówienia_to_Weryfikacja',
       'hours_Weryfikacja_to_Płatność',
       'hours_Złożenie zamówienia_to_Płatność', 'hours_Płatność_to_Wysyłka',
       'hours_Wysyłka_to_Dostawa', 'hours_Dostawa_to_Ocena',
       'hours_Dostawa_to_Zwrot', 'variant_name',
       'activities_count',
       'order_value', 'items', 'discount', 'region', 'source',]]

X_num = df_final[['hours_Złożenie zamówienia_to_Weryfikacja',
       'hours_Weryfikacja_to_Płatność',
       'hours_Złożenie zamówienia_to_Płatność', 'hours_Płatność_to_Wysyłka',
       'hours_Wysyłka_to_Dostawa', 'hours_Dostawa_to_Ocena',
       'hours_Dostawa_to_Zwrot', 'variant_name',
       'activities_count',
       'order_value', 'items', 'discount']]



X_cat = df_final[['region', 'source']]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_cat)
cat_feature_names = encoder.get_feature_names_out(X_cat.columns)

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


all_feature_names = list(cat_feature_names) + list(X_num.columns)

importances_forest = pd.Series(rf.feature_importances_, index=all_feature_names)\
             .sort_values(ascending=False)

print("TOP 10 Najważniejszych cech")
print(importances_forest.head(10).to_string())

##Widzimy o wiele lepsze accuracy niz przy poprzedniej probie bez szczegółowych cech pocesowych.
##TOP3 najwazniejsze cechy to czas spędzony pomiędzy aktywnościami.
##W szczególności widać, że elementem, który najbardziej wpływa na czas dostawy jest bezpośrednio logistyka


