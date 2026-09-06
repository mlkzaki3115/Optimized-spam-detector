from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_and_clean_data
from features import create_features
from train import train_models
from evaluate import evaluate_model
from sanity_checks import run_sanity_checks


DATA_PATH = ROOT / "data" / "spam_dataset.csv"
MODEL_PATH = ROOT / "models" / "spam_model.joblib"


def main():
    df = load_and_clean_data(DATA_PATH)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_vec,
        X_test_vec,
        vectorizer,
    ) = create_features(df)

    run_sanity_checks(
        df, X_train, X_test, y_train, y_test, X_train_vec
    )

    models, grid = train_models(X_train_vec, y_train)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(
            model, X_test_vec, y_test, name
        )

    print("Best parameters:", grid.best_params_)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": models["tuned_svm"],
            "vectorizer": vectorizer,
            "accuracy": results["tuned_svm"]["accuracy"],
        },
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()  