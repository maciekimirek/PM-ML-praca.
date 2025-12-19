import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pm4py

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


# Ladowanie danych


df = pd.read_csv("../../Data_source/Event_log.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")


# Cechy procesowe


def compute_hours(g):
    first_activity = [
        "Złożenie zamówienia", "Weryfikacja", "Złożenie zamówienia",
        "Płatność", "Wysyłka", "Dostawa", "Dostawa", "Dostawa", "Zwrot"
    ]
    second_activity = [
        "Weryfikacja", "Płatność", "Płatność",
        "Wysyłka", "Dostawa", "Ocena", "Zwrot", "Reklamacja", "Ocena"
    ]

    results = {}
    for a1, a2 in zip(first_activity, second_activity):
        try:
            ts1 = g.loc[g["Activity"] == a1, "Timestamp"].iloc[0]
            ts2 = g.loc[g["Activity"] == a2, "Timestamp"].iloc[0]
            results[f"hours_{a1}_to_{a2}"] = (ts2 - ts1).total_seconds() / 3600
        except IndexError:
            results[f"hours_{a1}_to_{a2}"] = np.nan

    return pd.Series(results)


df_cycletime = df.groupby("Order_ID").apply(
    compute_hours, include_groups=False
)


# Cechy procesowe cd (wyodrebnianie wariantow)


df = pm4py.format_dataframe(
    df,
    case_id="Order_ID",
    activity_key="Activity",
    timestamp_key="Timestamp"
)

rows = []
variant_counter = 1

for variant, subdf in pm4py.split_by_process_variant(df):
    for order_id in subdf["Order_ID"].unique():
        rows.append({
            "Order_ID": order_id,
            "variant_id": f"V{variant_counter}",
            "activities_count": len(variant)
        })
    variant_counter += 1

df_variants = pd.DataFrame(rows)


# Laczenie data frame


df_agg = df.groupby("Order_ID").agg(
    total_duration_hours=("Timestamp", lambda x: (x.max() - x.min()).total_seconds() / 3600),
    order_value=("Order_Value", "first"),
    items=("Items_Count", "first"),
    discount=("Discount", "first"),
    region=("Region", "first"),
    source=("Order_Source", "first")
).reset_index()

df_final = (
    df_cycletime
    .merge(df_variants, on="Order_ID")
    .merge(df_agg, on="Order_ID")
)


# Definiowanie klasyfikacji

def classify_duration(h):
    if h < 24:
        return "fast"
    if h < 48:
        return "medium"
    return "slow"

df_final["duration_class"] = df_final["total_duration_hours"].apply(classify_duration)

# Label encoding bo XGBOOST wymaga INT a nie STR
# Dalej drukujemy klase i indeks zeby pozniej latwiej zinterpretowac wyniki

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_final["duration_class"])

print("Mapowanie klas:")
for cls, idx in zip(label_encoder.classes_,
                    label_encoder.transform(label_encoder.classes_)):
    print(f"{cls} -> {idx}")


# Definicja cech


num_features = [
    "hours_Złożenie zamówienia_to_Weryfikacja",
    "hours_Weryfikacja_to_Płatność",
    "hours_Złożenie zamówienia_to_Płatność",
    "hours_Płatność_to_Wysyłka",
    "hours_Wysyłka_to_Dostawa",
    "hours_Dostawa_to_Ocena",
    "hours_Dostawa_to_Zwrot",
    "activities_count",
    "order_value",
    "items",
    "discount"
]

cat_features = ["region", "source", "variant_id"]

X = df_final[num_features + cat_features]


# Preprocessing


# Korzystamy z preprocesora, X numeryczne robia "passthrough"
# Dane kategoryczne sa kodowane a braki ignorowane
# Robimy to, a nie korzystamy tylko z onehotencodera
# bo encoder widzialby dane testowe i treningowe (data leakage)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)


# Train / test split


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42, stratify=y)


# Cross validation


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# RANDOM FOREST


pipe_rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

param_grid_rf = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [8, 12, 15],
    "model__min_samples_leaf": [3, 5, 9]
}

grid_rf = GridSearchCV(
    pipe_rf,
    param_grid_rf,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_


# XGBOOST


pipe_xgb = Pipeline([
    ("prep", preprocessor),
    ("model", xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ))
])

param_grid_xgb = {
    "model__n_estimators": [100, 200, 300,500],
    "model__max_depth": [6, 8, 10, 15],
    "model__learning_rate": [0.05, 0.1, 0.2]
}

grid_xgb = GridSearchCV(
    pipe_xgb,
    param_grid_xgb,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_


# LIGHTGBM


pipe_lgb = Pipeline([
    ("prep", preprocessor),
    ("model", lgb.LGBMClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1
    ))
])

param_grid_lgb = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__num_leaves": [20, 31, 50],
    "model__min_child_samples": [10,20, 30]
}

grid_lgb = GridSearchCV(
    pipe_lgb,
    param_grid_lgb,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_lgb.fit(X_train, y_train)
best_lgb = grid_lgb.best_estimator_


# Porownanie modeli

models = {
    "Random Forest": best_rf,
    "XGBoost": best_xgb,
    "LightGBM": best_lgb
}

print("\n" + "=" * 50)
print("POROWNANIE MODELI")
print("=" * 50)
print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 50)

for name, model in models.items():
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"{name:<20} {train_acc:<12.4f} {test_acc:<12.4f}")

# ======================================================================
# Confusion matrix + raport (dla najlepszego modelu)
# ======================================================================

best_model = best_lgb  # mozna zmienic recznie

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

print(f"\nCONFUSION MATRIX dla:")
print(cm_df)

print(f"\nCLASSIFICATION REPORT dla:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_lgb, X, y, cv=5)
print(scores.mean(), scores.std())


# pobranie preprocessora
prep = best_lgb.named_steps["prep"]
model = best_lgb.named_steps["model"]

# nazwy cech po transformacji
num_names = num_features
cat_names = prep.named_transformers_["cat"] \
                .get_feature_names_out(cat_features)

feature_names = np.concatenate([num_names, cat_names])

# importance
importances = model.feature_importances_

fi = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    .sort_values("importance", ascending=False)
)
#cechy
print(fi.head(15))

#Wykres
plt.figure(figsize=(10, 6))
plt.barh(fi["feature"], fi["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Feature importance")
plt.title("Top 15 najważniejszych cech – LightGBM")
plt.tight_layout()
plt.show()