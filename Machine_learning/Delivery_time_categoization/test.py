import pandas as pd


import pandas as pd
import numpy as np
from numpy.ma.core import transpose
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

df = pd.DataFrame({
    "city": np.random.choice(
        ["Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań"],
        size=60
    ),

    "channel": np.random.choice(
        ["online", "shop", "mobile_app"],
        size=60
    ),

    "payment_method": np.random.choice(
        ["card", "blik", "cash", "transfer"],
        size=60
    ),

    "age": np.random.randint(18, 70, size=60),

    "orders_last_30d": np.random.poisson(lam=3, size=60),

    "order_value": np.round(
        np.random.uniform(50, 600, size=60), 2
    ),

    "discount": np.round(
        np.random.choice([0, 5, 10, 15, 20], size=60), 1
    ),

    "delivery_time_days": np.random.randint(1, 10, size=60),

    "klasa_klienta": np.random.choice(
        ["biedny", "sredni", "zamozny"],
        p=[0.3, 0.4, 0.3],
        size=60
    )
})

print(df.head(3).to_string())
print(df.info())

#Podzial na X i Y

y = df["klasa_klienta"]
X = df.drop(columns="klasa_klienta")
print(X.head(3).to_string())

##Transformacja kolumn ktore sa obcjectami

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer(
    transformers=[
        ("city", OneHotEncoder(handle_unknown= "ignore"), ["city"]),
        ("channel", OneHotEncoder(handle_unknown= "ignore"), ["channel"]),
        ("payment_method", OneHotEncoder(handle_unknown= "ignore"), ["payment_method"])
    ], remainder= "passthrough"
)

## nic dlaj nie robimy, zajmiemy sie tym w grid searchu

## Zamiana naszego targetu

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print(y)

## Pipeline

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

pip = Pipeline(
    steps=[
        ("preprocesing", ct),
        ("model", model)
    ]
)

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
#
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state= 42)
#
# pip.fit(X_train,y_train)
# y_pred = pip.predict(X_test)
#
# print(accuracy_score(y_test,y_pred))
# print(pd.DataFrame(confusion_matrix(y_test,y_pred)))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# for k in pip.get_params().keys():
#     print(k)

parameters_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [8, 12],
    "model__min_samples_leaf": [3, 5]
}

grid_search = GridSearchCV(
    pip,
    parameters_grid
)

grid_search.fit(X,y)

print(f'BEST PARAMS{grid_search.best_params_}')
print(f'BEST ESTIMATORS{grid_search.best_estimator_}')
print(f'BEST ESTIMATORS{grid_search.best_score_}')
