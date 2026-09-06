from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, name):
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    return {
        "name": name,
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }