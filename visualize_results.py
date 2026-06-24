import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



df = pd.read_csv("data/processed/cleaned_delivery_data.csv")

X = df.drop("Delivery_Time_Hours", axis=1)
y = df["Delivery_Time_Hours"]

categorical = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        )
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators":[100,200],
    "model__max_depth":[10,20,None]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(X_train,y_train)

best_model = grid.best_estimator_

predictions = best_model.predict(X_test)



mae = mean_absolute_error(y_test,predictions)
rmse = np.sqrt(mean_squared_error(y_test,predictions))
r2 = r2_score(y_test,predictions)

print("Random Forest Results")
print("--------------------------")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")


plt.figure(figsize=(7,6))

plt.scatter(
    y_test,
    predictions,
    alpha=.7
)

plt.plot(
    [y_test.min(),y_test.max()],
    [y_test.min(),y_test.max()],
    "r--"
)

plt.title("Actual vs Predicted Delivery Time")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.tight_layout()

plt.savefig("visuals/actual_vs_predicted.png")

residuals = y_test-predictions

plt.figure(figsize=(7,6))

plt.scatter(
    predictions,
    residuals,
    alpha=.7
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.title("Residual Analysis")

plt.tight_layout()

plt.savefig("visuals/residual_plot.png")


encoded_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = best_model.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature":encoded_names,
    "Importance":importances
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Feature"][:10],
    importance_df["Importance"][:10]
)

plt.gca().invert_yaxis()

plt.title("Top 10 Most Important Features")

plt.tight_layout()

plt.savefig("visuals/feature_importance.png")


results = pd.DataFrame({

    "Model":[
        "Random Forest"
    ],

    "MAE":[mae],

    "RMSE":[rmse],

    "R2":[r2]

})

plt.figure(figsize=(6,5))

plt.bar(
    results["Model"],
    results["R2"]
)

plt.ylabel("R² Score")

plt.title("Model Performance")

plt.tight_layout()

plt.savefig("visuals/model_comparison.png")

print("\nGraphs saved successfully!")