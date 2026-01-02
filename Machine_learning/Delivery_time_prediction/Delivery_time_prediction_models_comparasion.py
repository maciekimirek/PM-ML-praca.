##### N_ITER USTAWIONE na 1 - DO ZMIANY na PPROD

import pandas as pd
import numpy as np

from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.ensemble import  RandomForestRegressor

import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# 1. Wczytywanie danaych

df = pd.read_csv("../../Data_source/Event_log.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed")

# 2. Budowa datasetu prefixowego

# rows = []
#
# for order_id, g in df.groupby("Order_ID"):
#     g = g.sort_values("Timestamp").reset_index(drop=True)
#
#     start_time = g.loc[0, "Timestamp"]
#     end_time = g.loc[len(g) - 1, "Timestamp"]
#
#
#     for i in range(1, len(g)):
#         current_time = g.loc[i, "Timestamp"]
#
#         rows.append({
#             "Order_ID": order_id,
#
#
#             "current_activity": g.loc[i, "Activity"],
#             "elapsed_time_hours": (current_time - start_time).total_seconds() / 3600,
#             "steps_done": i,
#             "last_step_duration": (
#                 (g.loc[i, "Timestamp"] - g.loc[i - 1, "Timestamp"]).total_seconds() / 3600
#             ),
#
#
#             "remaining_time_hours": (
#                 (end_time - current_time).total_seconds() / 3600
#             )
#         })

rows = []

for order_id, g in df.groupby("Order_ID"):
    g = g.sort_values("Timestamp").reset_index(drop=True)

    start_time = g.loc[0, "Timestamp"]

    durations = []

    for i in range(1, len(g)):
        current_time = g.loc[i, "Timestamp"]
        step_duration = (
            g.loc[i, "Timestamp"] - g.loc[i - 1, "Timestamp"]
        ).total_seconds() / 3600

        durations.append(step_duration)

        rows.append({
            "Order_ID": order_id,
            "current_activity": g.loc[i, "Activity"],

            # PRODUKCYJNE
            "elapsed_time_hours": (current_time - start_time).total_seconds() / 3600,
            "steps_done": i,
            "last_step_duration": step_duration,
            "avg_step_duration": np.mean(durations),
            "max_step_duration": np.max(durations),

            # TARGET (tylko do treningu)
            "remaining_time_hours": (
                g.loc[len(g)-1, "Timestamp"] - current_time
            ).total_seconds() / 3600
        })

df_ml = pd.DataFrame(rows)


# 3. Dodanie reszty cech

df_static = df.groupby("Order_ID").agg(
    order_value=('Order_Value', 'first'),
    items=('Items_Count', 'first'),
    discount=('Discount', 'first'),
    region=('Region', 'first'),
    source=('Order_Source', 'first')
).reset_index()

# 3.1 Dodatkowe cechy procesowe  TEGO NIE MOGE UZYC NA PRODUKCJI
#
# df_ml["progress_ratio"] = df_ml["steps_done"] / (
#     df.groupby("Order_ID")["Activity"].transform("count") - 1
# )

## Dodanie ponizszego step duratin trend tylko pogarsza wyniki

# df_ml["step_duration_trend"] = (
#     df_ml["last_step_duration"] / df_ml["avg_step_duration"]
# )


df_ml = df_ml.merge(df_static, on="Order_ID", how="left")


# 4. Podzial na dane numeryczne i kategoyczne

X_num = [
        "elapsed_time_hours",
        "steps_done",
        "last_step_duration",
        "order_value",
        "items",
        "discount"
    ]


X_cat = [
        "current_activity",
        "region",
        "source"
    ]


# Preprocesing

Preprocesing = ColumnTransformer(
    transformers=[
        ("num", "passthrough", X_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output= False), X_cat)
    ]
)

X = df_ml[X_num + X_cat]
y = df_ml["remaining_time_hours"]

#  5. Split danych korzystamy z GroupShufffleSplit aby uniknac data lekage.


from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(
    gss.split(X, y, groups=df_ml["Order_ID"])
)

# y_log = np.log1p(y) danie nie sa "ognowate" to tylko pogarsza wynik
X_train, X_test ,y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]




#  6. Przygotowanie Parametow

param_random_lgb = {
    "model__n_estimators": randint(400, 2000),
    "model__learning_rate": loguniform(5e-4, 0.05),
    "model__num_leaves": randint(31, 255),
    "model__max_depth": randint(5, 20),
    "model__min_child_samples": randint(20, 150),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4)
}



param_random_xgb = {
    "model__n_estimators": randint(400, 2000),
    "model__learning_rate": loguniform(5e-4, 0.05),
    "model__max_depth": randint(3, 10),
    "model__min_child_weight": randint(5, 50),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4)
}



param_random_forest = {
    "model__n_estimators": randint(300, 1500),
    "model__max_depth": randint(5, 25),
    "model__min_samples_split": randint(10, 100),
    "model__min_samples_leaf": randint(5, 50),
    "model__max_features": uniform(0.4, 0.6)
}


# 7 MODELE

rf_model = RandomForestRegressor(random_state= 42)
xgb_model = xgb.XGBRegressor(random_state=42, tree_method="hist", objective="reg:squarederror")
lgb_model = lgb.LGBMRegressor(random_state= 42, verbosity = -1)



# 7 PIPELINES

#RandomForest

pip_rf = Pipeline(
        steps=
        [
            ("prep", Preprocesing),
            ("model", rf_model )
        ]
    )

# XGB

pip_xgb = Pipeline(
        steps=
        [
            ("prep", Preprocesing),
            ("model", xgb_model )
        ]
    )

# LGB

pip_lgb = Pipeline(
        steps=
        [
            ("prep", Preprocesing),
            ("model", lgb_model )
        ]
    )

# 8 RandomizerSearch

#RandomForest

rf_rs = RandomizedSearchCV(
    estimator= pip_rf,
    param_distributions= param_random_forest,
    n_iter = 1,
    n_jobs = -1,
    cv = GroupKFold(n_splits=5),
    scoring="neg_mean_absolute_error"
)

rf_rs.fit(
    X_train,
    y_train,
    groups=df_ml.loc[X_train.index, "Order_ID"]
)

#XGB

xgb_rs = RandomizedSearchCV(
    estimator= pip_xgb,
    param_distributions= param_random_xgb,
    n_iter = 1,
    n_jobs = -1,
    cv = GroupKFold(n_splits=5),
    scoring="neg_mean_absolute_error"

)

xgb_rs.fit(
    X_train,
    y_train,
    groups=df_ml.loc[X_train.index, "Order_ID"]
)

#LGB

lgb_rs = RandomizedSearchCV(
    estimator= pip_lgb,
    param_distributions= param_random_lgb,
    n_iter = 1,
    n_jobs = -1,
    cv = GroupKFold(n_splits=5),
    scoring="neg_mean_absolute_error"

)

lgb_rs.fit(
    X_train,
    y_train,
    groups=df_ml.loc[X_train.index, "Order_ID"]
)

# 9. Ewaluacja

models = [rf_rs, lgb_rs, xgb_rs]


for rs in models:
    best_model = rs.best_estimator_

    y_pred = best_model.predict(X_test)


    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{type(best_model.named_steps['model']).__name__}")
    print(f"MAE (hours): {mae:.2f}")
    print(f"RMSE (hours): {rmse:.2f}")
    print("-" * 40)

    plt.figure(figsize=(10, 6))

    plt.scatter(y_test, y_pred, alpha=0.4, color="#4a76d4")

    plt.plot([0, max(y_test)],[0, max(y_test)],color="#d62728",linewidth=2)

    plt.xlabel("Rzeczywisty Remaining Time (h)")
    plt.ylabel("Prognozowany Remaining Time (h)")
    plt.title(f' Skutecznosc modelu {type(best_model.named_steps["model"]).__name__}')

    plt.grid(alpha=0.2)
    plt.show()

# 11. Feature Importance — wybór najlepszego modelu po MAE

results = []

for rs in [rf_rs, lgb_rs, xgb_rs]:
    model = rs.best_estimator_
    name = type(model.named_steps["model"]).__name__

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    results.append((mae, name, model))

# wybieramy najmniejsze MAE
best_mae, best_name, best_model = min(results, key=lambda x: x[0])

print(f"Najlepszy model: {best_name}")
print(f"MAE: {best_mae:.2f}")

# Ferature Importance

feature_names = best_model.named_steps["prep"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nWażność cech:")
print(fi.head(20).to_string(index=False))

# 12. Wykres waznosci cech

fi.head(20).plot(kind="barh", x="feature", y="importance")
plt.title(f' Feature Importance dla {best_name}')
plt.gca().invert_yaxis()
plt.show()






