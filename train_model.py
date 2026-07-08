"""
train_model.py

Loads the post dataset, runs EDA, engineers features, trains and compares
three regression models to predict engagement_rate, and saves:
  - images/*.png                  (EDA + evaluation charts)
  - models/model.pkl              (best trained model)
  - models/feature_columns.json   (column order the model expects)
  - results.json                  (metrics used in the case study)

Usage:
    python3 train_model.py
    python3 train_model.py --data data/social_media_posts.csv
"""

import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

sns.set_style("whitegrid")

DROP_COLS = ["post_id", "likes", "comments", "shares", "engagement_rate"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the engagement forecasting model.")
    parser.add_argument("--data", default="data/social_media_posts.csv", help="path to input CSV")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the train/test split")
    return parser.parse_args()


def hour_bucket(h):
    if 0 <= h <= 6:
        return "early_morning"
    if 7 <= h <= 11:
        return "morning"
    if 12 <= h <= 17:
        return "afternoon"
    if 18 <= h <= 22:
        return "evening"
    return "late_night"


def engineer_features(data: pd.DataFrame):
    d = data.copy()
    d["hour_bucket"] = d["hour_of_day"].apply(hour_bucket)
    d = pd.get_dummies(d, columns=["platform", "post_type", "day_of_week", "hour_bucket"], drop_first=True)
    feature_cols = [c for c in d.columns if c not in DROP_COLS]
    return d, feature_cols


def plot_eda(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.boxplot(data=df, x="post_type", y="engagement_rate", ax=axes[0],
                order=["video", "carousel", "image", "text"])
    axes[0].set_title("Engagement rate by post type")
    axes[0].set_ylabel("Engagement rate (%)")

    hourly = df.groupby("hour_of_day")["engagement_rate"].mean()
    axes[1].plot(hourly.index, hourly.values, marker="o")
    axes[1].set_title("Average engagement rate by hour of day")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Engagement rate (%)")

    numeric_cols = ["engagement_rate", "hashtag_count", "caption_length", "follower_count", "hour_of_day"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[2], cbar=False)
    axes[2].set_title("Correlation between numeric features")

    plt.tight_layout()
    plt.savefig("images/eda_overview.png", dpi=130)
    plt.close()


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }, pred


def plot_predicted_vs_actual(y_test, pred, model_name):
    plt.figure(figsize=(5.5, 5))
    plt.scatter(y_test, pred, alpha=0.4, s=18)
    lims = [0, max(y_test.max(), pred.max()) * 1.05]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual engagement rate (%)")
    plt.ylabel("Predicted engagement rate (%)")
    plt.title(f"Predicted vs actual -- {model_name}")
    plt.tight_layout()
    plt.savefig("images/predicted_vs_actual.png", dpi=130)
    plt.close()


def plot_feature_importance(model, feature_cols):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(6, 5))
    importances[::-1].plot(kind="barh", color="#3b6fa0")
    plt.title("Top 10 feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=130)
    plt.close()
    return importances.to_dict()


def main():
    args = parse_args()
    df = pd.read_csv(args.data)

    plot_eda(df)

    df_fe, feature_cols = engineer_features(df)
    X, y = df_fe[feature_cols], df_fe["engagement_rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    lr = LinearRegression().fit(X_train, y_train)
    lr_metrics, lr_pred = evaluate(lr, X_test, y_test)

    gb = GradientBoostingRegressor(random_state=args.seed).fit(X_train, y_train)
    gb_metrics, gb_pred = evaluate(gb, X_test, y_test)

    # grid search over depth/estimators for the random forest
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=args.seed),
        param_grid={"n_estimators": [100, 200], "max_depth": [6, 10, None]},
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    ).fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_metrics, rf_pred = evaluate(rf_best, X_test, y_test)
    rf_metrics["best_params"] = rf_search.best_params_
    rf_cv_scores = cross_val_score(rf_best, X, y, cv=5, scoring="r2")
    rf_metrics["cv_r2_mean"] = float(rf_cv_scores.mean())
    rf_metrics["cv_r2_std"] = float(rf_cv_scores.std())

    print("Linear Regression:", lr_metrics)
    print("Gradient Boosting:", gb_metrics)
    print("Random Forest:     ", rf_metrics)

    candidates = {
        "linear_regression": (lr, lr_metrics, lr_pred),
        "gradient_boosting": (gb, gb_metrics, gb_pred),
        "random_forest": (rf_best, rf_metrics, rf_pred),
    }
    winner_name = min(candidates, key=lambda k: candidates[k][1]["rmse"])
    winner_model, winner_metrics, winner_pred = candidates[winner_name]

    joblib.dump(winner_model, "models/model.pkl")
    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    plot_predicted_vs_actual(y_test, winner_pred, winner_name)

    if hasattr(winner_model, "feature_importances_"):
        top_features = plot_feature_importance(winner_model, feature_cols)
    else:
        top_features = {}

    test_out = X_test.copy()
    test_out["actual_engagement_rate"] = y_test.values
    test_out["predicted_engagement_rate"] = winner_pred
    test_out.to_csv("data/test_predictions.csv", index=False)

    results = {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "linear_regression": lr_metrics,
        "gradient_boosting": gb_metrics,
        "random_forest": rf_metrics,
        "winner": winner_name,
        "top_features": top_features,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWinner: {winner_name}")
    print("Saved model, charts, and results.json")


if __name__ == "__main__":
    main()
