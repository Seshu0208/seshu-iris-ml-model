from iris_ml_model import load_data, train_model, evaluate_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def test_data_loading():

    df = load_data()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_no_missing_values():
    
    df = load_data()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_model_training():

    model, _, _ = train_model()

    assert isinstance(model, KNeighborsClassifier)


def test_prediction_shape():

    model, X_test, _ = train_model()
    preds = model.predict(X_test)

    assert len(preds) == len(X_test)


def test_prediction_classes():

    model, X_test, _ = train_model()
    preds = model.predict(X_test)

    assert set(preds).issubset({0, 1, 2})


def test_model_accuracy():

    model, X_test, y_test = train_model()
    accuracy, _ = evaluate_model(model, X_test, y_test)

    assert accuracy >= 0.9