import os
import gzip
import json
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def limpiarDatos(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.dropna()
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x >= 4 else x).astype('category')
    x, y = df.drop(columns=['default']), df['default']
    return df, x, y


def pipeline() -> Pipeline:
    caracteristicas = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[('categoria', OneHotEncoder(), caracteristicas)],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline


def hiperParametros(pipeline, x, y):
    parametros = {
        'classifier__n_estimators': [200],
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [10],
        'classifier__min_samples_leaf': [1, 2],
    }
    gridSearch = GridSearchCV(
        pipeline,
        parametros,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2,
        refit=True
    )
    return gridSearch.fit(x, y)


def guardar(model):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as file:
        pickle.dump(model, file)


def metricas(pipeline, x_train, y_train, x_test, y_test):
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    metricasTrain = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred))
    }

    metricasTest = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred))
    }

    return metricasTrain, metricasTest


def matrizConfusion(pipeline, x_train, y_train, x_test, y_test):
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    cm_metrics_train = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    }

    cm_metrics_test = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    }

    return cm_metrics_train, cm_metrics_test


def guardarMetricas(metrics_train, metrics_test, cm_metrics_train, cm_metrics_test, file_path="files/output/metrics.json"):
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    metricas = [metrics_train, metrics_test, cm_metrics_train, cm_metrics_test]

    with open(file_path, "w") as f:
        for i in metricas:
            f.write(json.dumps(i) + "\n")


# Cargar datos
test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

# Limpiar datos
test, x_test, y_test = limpiarDatos(test)
train, x_train, y_train = limpiarDatos(train)

# Crear y entrenar el modelo
modelo = pipeline()
modelo = hiperParametros(modelo, x_train, y_train)

# Guardar el modelo
guardar(modelo)

# Calcular métricas
metrics_train, metrics_test = metricas(modelo, x_train, y_train, x_test, y_test)
cm_metrics_train, cm_metrics_test = matrizConfusion(modelo, x_train, y_train, x_test, y_test)

# Guardar métricas
guardarMetricas(metrics_train, metrics_test, cm_metrics_train, cm_metrics_test)