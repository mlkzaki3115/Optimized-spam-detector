from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def create_features(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_vec, X_test_vec, vectorizer