import pandas as pd
import pm4py
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb


df = pd.read_csv("../../Data_source/Event_log.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed")

# Wyodrebnianie cech procesowych
# Tworzymy funkcje, bo możemy mieć elementy, które nie maja danych aktywności.

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
            hours = np.nan

        colname = f"hours_{a1}_to_{a2}"
        results[colname] = hours

    return pd.Series(results)


# Pierwsza czesc dataframe
df_cycletime = (
    df.groupby("Order_ID").apply(compute_hours, include_groups=False)
)

# Wyodrębnienie wariantów procesu
df = pm4py.format_dataframe(df, case_id='Order_ID', activity_key='Activity', timestamp_key='Timestamp')

rows = []
variant_counter = 1

for variant, subdf in pm4py.split_by_process_variant(df):
    variant_id = f"V{variant_counter}"

    for order_id in subdf["Order_ID"].unique():
        rows.append({
            "Order_ID": order_id,
            "variant_original": variant,
            "variant_id": variant_id
        })

    variant_counter += 1


df_variants = pd.DataFrame(rows)
df_variants["activities_count"] = df_variants["variant_original"].apply(len)

df_merged = df_cycletime.merge(df_variants, on="Order_ID", how="left")


# Tworzenie DF z poczatkowymi cechami
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


# Laczenie
df_final = df_merged.merge(df_agg, on="Order_ID", how="left")


def classify_duration(h):
    if h < 24:
        return "fast"
    if h < 48:
        return "medium"
    else:
        return "slow"


df_final['duration_class'] = df_final['total_duration_hours'].apply(classify_duration)

y = df_final['duration_class']

X_num = df_final[[
    'hours_Złożenie zamówienia_to_Weryfikacja',
    'hours_Weryfikacja_to_Płatność',
    'hours_Złożenie zamówienia_to_Płatność',
    'hours_Płatność_to_Wysyłka',
    'hours_Wysyłka_to_Dostawa',
    'hours_Dostawa_to_Ocena',
    'hours_Dostawa_to_Zwrot',
    'activities_count',
    'order_value', 'items', 'discount'
]]

X_cat = df_final[['region', 'source', 'variant_id']]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(X_cat)
cat_feature_names = encoder.get_feature_names_out(X_cat.columns)

X_final = np.hstack([X_cat_encoded, X_num])

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)


# LightGBM
lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.05,
    random_state=42,
    class_weight='balanced'
)

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

print("LightGBM — accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=lgbm.classes_, columns=lgbm.classes_))


all_feature_names = list(cat_feature_names) + list(X_num.columns)

importances_lgbm = pd.Series(
    lgbm.feature_importances_,
    index=all_feature_names
).sort_values(ascending=False)

print("TOP 10 Najważniejszych cech")
print(importances_lgbm.head(10).to_string())

