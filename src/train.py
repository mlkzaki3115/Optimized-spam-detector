from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_models(X_train, y_train):
    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "svm": LinearSVC(),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    svm_grid = GridSearchCV(
        LinearSVC(),
        {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10, 50]},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    svm_grid.fit(X_train, y_train)

    models["tuned_svm"] = svm_grid.best_estimator_
    return models, svm_grid