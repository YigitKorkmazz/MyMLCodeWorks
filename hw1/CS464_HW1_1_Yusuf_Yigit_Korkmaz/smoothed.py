from main import load_data, run_model


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data()
    run_model(X_train, y_train, X_val, y_val, alpha=1)
