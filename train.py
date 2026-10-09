import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42


def prepare_data():
    print("Fetching and splitting California Housing data...")
    X, y = fetch_california_housing(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def execute_run(run_name, max_depth, learning_rate, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=run_name):
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": SEED,
        }
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False,
        )

        preds = model.predict(dval)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))

        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        print(f"{run_name} | depth={max_depth} lr={learning_rate} | "
              f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        return {"run": run_name, "max_depth": max_depth, "learning_rate": learning_rate,
                "rmse": rmse, "mae": mae, "r2": r2}


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("California_Housing_Optimization")

    X_train, X_val, y_train, y_val = prepare_data()

    run_configs = [
        {"name": "Run 1", "depth": 3, "lr": 0.1},
        {"name": "Run 2", "depth": 5, "lr": 0.05},
        {"name": "Run 3", "depth": 7, "lr": 0.01},
    ]

    results = [execute_run(c["name"], c["depth"], c["lr"],
                           X_train, X_val, y_train, y_val) for c in run_configs]

    best = min(results, key=lambda r: r["rmse"])
    print(f"\nBest model (lowest RMSE): {best['run']} -> RMSE={best['rmse']:.4f}")


if __name__ == "__main__":
    main()