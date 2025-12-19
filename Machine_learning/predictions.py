import pandas as pd
import numpy as np
import pm4py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb

# =========================
# 1. Wczytanie danych
# =========================

df = pd.read_csv("../Data_source/Event_log.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="mixed")

# =========================
# 2. Budowa datasetu PREFIXOWEGO
#    (kluczowa zmiana!)
# =========================

rows = []

for order_id, g in df.groupby("Order_ID"):
    g = g.sort_values("Timestamp").reset_index(drop=True)

    start_time = g.loc[0, "Timestamp"]
    end_time = g.loc[len(g) - 1, "Timestamp"]

    for i in range(1, len(g)):  # prefix >= 1 zdarzenie
        current_time = g.loc[i, "Timestamp"]

        rows.append({
            "Order_ID": order_id,

            # ===== CECHY PROCESOWE (PRZESZŁOŚĆ) =====
            "current_activity": g.loc[i, "Activity"],
            "elapsed_time_hours": (current_time - start_time).total_seconds() / 3600,
            "steps_done": i,
            "last_step_duration": (
                (g.loc[i, "Timestamp"] - g.loc[i - 1, "Timestamp"]).total_seconds() / 3600
            ),

            # ===== TARGET =====
            "remaining_time_hours": (
                (end_time - current_time).total_seconds() / 3600
            )
        })

df_ml = pd.DataFrame(rows)


# =========================
# 3. Dodanie cech statycznych (biznesowych)
# =========================

df_static = df.groupby("Order_ID").agg(
    order_value=('Order_Value', 'first'),
    items=('Items_Count', 'first'),
    discount=('Discount', 'first'),
    region=('Region', 'first'),
    source=('Order_Source', 'first')
).reset_index()

df_ml = df_ml.merge(df_static, on="Order_ID", how="left")

print(df_ml.head(4).to_string())
#
# # =========================
# # 4. Przygotowanie X / y
# # =========================
#
# X_num = df_ml[
#     [
#         "elapsed_time_hours",
#         "steps_done",
#         "last_step_duration",
#         "order_value",
#         "items",
#         "discount"
#     ]
# ]
#
# X_cat = df_ml[
#     [
#         "current_activity",
#         "region",
#         "source"
#     ]
# ]
#
# encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# X_cat_encoded = encoder.fit_transform(X_cat)
# cat_feature_names = encoder.get_feature_names_out(X_cat.columns)
#
# X = np.hstack([X_cat_encoded, X_num])
# y = df_ml["remaining_time_hours"]
#
# all_feature_names = list(cat_feature_names) + list(X_num.columns)
#
# # =========================
# # 5. Split (bez leakage)
# # =========================
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )
#
# # =========================
# # 6. Model — REGRESJA
# # =========================
#
# model = lgb.LGBMRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     num_leaves=31,
#     random_state=42,
#     n_jobs=-1
# )
#
# model.fit(X_train, y_train)
#
# # =========================
# # 7. Ewaluacja
# # =========================
#
# y_pred = model.predict(X_test)
#
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#
#
# print(f"MAE (hours): {mae:.2f}")
# print(f"RMSE (hours): {rmse:.2f}")
#
# # =========================
# # 8. Feature importance
# # =========================
#
# importances = pd.Series(
#     model.feature_importances_,
#     index=all_feature_names
# ).sort_values(ascending=False)
#
# print("\nTOP 10 najważniejszych cech:")
# print(importances.head(10).to_string())
