def run_sanity_checks(df, X_train, X_test, y_train, y_test, X_train_vec):
    """Verify data integrity, shapes, and isolation before modeling."""
    # Ensure labels strictly match binary targets
    assert set(df['label'].unique()) <= {"spam", "ham"}, "Unexpected label values detected in target column."
    
    # Ensure vectorizer rows match label length
    assert X_train_vec.shape[0] == len(y_train), "Mismatch between feature rows and training labels."
    
    # Ensure zero index overlap between train and test sets
    overlap = set(X_train.index).intersection(set(X_test.index))
    assert len(overlap) == 0, "Data leakage detected: Index overlap between train and test splits."
    
    print("[PASS] All automated sanity checks passed successfully.")