"""
Classificação de risco de readmissão hospitalar em 30 dias.

Uso Random Forest como modelo principal. Testei também regressão logística
como referência mais simples e interpretável — o comparativo entre os dois
está no notebook 04. A Random Forest venceu em recall sem perder muito em
precisão, e para esse problema prefiro errar para o lado de sinalizar risco
demais do que de menos: um falso negativo aqui significa um paciente de
alto risco saindo do hospital sem nenhum acompanhamento programado.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_PATH = Path(__file__).parents[2] / 'models'
MODELS_PATH.mkdir(exist_ok=True)

COLUNAS_ID = ['internacao_id', 'readmissao_30d']


def separar_treino_teste(features: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> Tuple:
    """
    Separa treino e teste de forma estratificada pela variável alvo.

    Estratifico porque a readmissão é a classe minoritária — sem
    estratificação, um split ruim de sorte poderia deixar o teste com uma
    proporção de casos positivos bem diferente do treino, distorcendo as
    métricas de avaliação.
    """
    X = features.drop(columns=COLUNAS_ID)
    y = features['readmissao_30d']

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_treino, X_teste, y_treino, y_teste


def treinar_random_forest(X_treino: pd.DataFrame, y_treino: pd.Series, seed: int = 42) -> RandomForestClassifier:
    """
    Treina a Random Forest com balanceamento de classe.

    class_weight='balanced' compensa o desbalanceamento sem eu precisar
    fazer oversampling manual — a árvore passa a penalizar mais o erro na
    classe minoritária (readmissão) durante o treinamento.
    """
    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
    )
    modelo.fit(X_treino, y_treino)
    return modelo


def treinar_logistic_baseline(X_treino: pd.DataFrame, y_treino: pd.Series, seed: int = 42) -> Tuple:
    """
    Baseline de regressão logística, usado como referência de interpretabilidade.

    Padronizo as variáveis numéricas porque a regressão logística é
    sensível à escala — sem isso, idade e valor_total (que estão em
    escalas bem diferentes) distorceriam os coeficientes.
    """
    scaler = StandardScaler()
    X_treino_esc = scaler.fit_transform(X_treino)

    modelo = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
    modelo.fit(X_treino_esc, y_treino)
    return modelo, scaler


def avaliar_modelo(modelo, X_teste: pd.DataFrame, y_teste: pd.Series, scaler=None) -> Dict:
    """
    Avalia o modelo no conjunto de teste.

    Reporto AUC, recall e precisão separadamente em vez de só acurácia —
    com ~20% de positivos, um modelo que chuta "não readmite" para todo
    mundo já acerta 80% e não serve para nada na prática.
    """
    X_avaliar = scaler.transform(X_teste) if scaler is not None else X_teste

    y_pred = modelo.predict(X_avaliar)
    y_prob = modelo.predict_proba(X_avaliar)[:, 1]

    metricas = {
        'auc': roc_auc_score(y_teste, y_prob),
        'f1': f1_score(y_teste, y_pred),
        'recall': recall_score(y_teste, y_pred),
        'precisao': precision_score(y_teste, y_pred),
        'matriz_confusao': confusion_matrix(y_teste, y_pred).tolist(),
    }
    return metricas


def extrair_importancia_features(modelo: RandomForestClassifier, colunas: list) -> pd.DataFrame:
    """Retorna a importância de cada feature, ordenada da mais para a menos relevante."""
    importancia = pd.DataFrame({
        'feature': colunas,
        'importancia': modelo.feature_importances_,
    }).sort_values('importancia', ascending=False).reset_index(drop=True)
    return importancia


if __name__ == '__main__':
    from src.data.load_data import pipeline_completo
    from src.features.engenharia_features import preparar_features_readmissao

    internacoes, _ = pipeline_completo()
    features = preparar_features_readmissao(internacoes)

    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(features)
    modelo = treinar_random_forest(X_treino, y_treino)
    metricas = avaliar_modelo(modelo, X_teste, y_teste)

    logger.info(f'AUC: {metricas["auc"]:.3f} | Recall: {metricas["recall"]:.3f} | Precisão: {metricas["precisao"]:.3f}')
