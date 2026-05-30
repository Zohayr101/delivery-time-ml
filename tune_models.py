
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/cleaned_delivery_data.csv")

X = df.drop("Delivery_Time_Hours", axis=1)
y = df["Delivery_Time_Hours"]

categorical = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20]
}

rf_search = GridSearchCV(rf_pipeline, rf_grid, cv=5, scoring="r2")
rf_search.fit(X_train, y_train)

gb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

gb_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1]
}

gb_search = GridSearchCV(gb_pipeline, gb_grid, cv=5, scoring="r2")
gb_search.fit(X_train, y_train)

for name, model in {
    "Random Forest": rf_search.best_estimator_,
    "Gradient Boosting": gb_search.best_estimator_
}.items():
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
