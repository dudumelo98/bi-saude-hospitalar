"""
Clusterização de hospitais por perfil operacional.

A ideia aqui é a parte de "otimização de recursos" do escopo do projeto:
em vez de tratar cada hospital individualmente, agrupo unidades com perfil
parecido de ocupação, permanência e readmissão para que a rede possa
aplicar a mesma estratégia de gestão a cada grupo — redistribuição de
leitos, reforço de equipe ou programa de acompanhamento pós-alta — sem
precisar de um plano sob medida para cada um dos hospitais.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLUNAS_CLUSTER = [
    'tempo_medio_permanencia', 'taxa_readmissao', 'taxa_mortalidade',
    'taxa_ocupacao_media', 'rotatividade_leitos'
]


def preparar_dados_cluster(indicadores_hospital: pd.DataFrame) -> tuple:
    """Padroniza os indicadores antes do K-Means — sem isso, ocupação (0-1)
    e valor médio de internação (milhares de reais) teriam pesos muito
    diferentes na distância euclidiana usada pelo algoritmo."""
    X = indicadores_hospital[COLUNAS_CLUSTER].copy()
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)
    return X_escalado, scaler


def escolher_k(X_escalado: np.ndarray, k_min: int = 2, k_max: int = 8, seed: int = 42) -> pd.DataFrame:
    """
    Testa uma faixa de K e retorna inércia e silhouette de cada um.

    Não decido o K só pelo cotovelo da inércia — em datasets pequenos como
    esse (40 hospitais), a curva de inércia às vezes não tem um cotovelo
    nítido, então o silhouette score serve de segundo critério para
    confirmar a escolha.
    """
    resultados = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = modelo.fit_predict(X_escalado)
        resultados.append({
            'k': k,
            'inercia': modelo.inertia_,
            'silhouette': silhouette_score(X_escalado, labels),
        })
    return pd.DataFrame(resultados)


def treinar_kmeans(X_escalado: np.ndarray, k: int, seed: int = 42) -> KMeans:
    """Treina o K-Means final com o K escolhido."""
    modelo = KMeans(n_clusters=k, random_state=seed, n_init=10)
    modelo.fit(X_escalado)
    return modelo


def descrever_clusters(indicadores_hospital: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Devolve a média de cada indicador por cluster — é isso que eu leio
    para dar um nome de negócio a cada grupo (ex: "alta ocupação crônica")."""
    df = indicadores_hospital.copy()
    df['cluster'] = labels
    perfil = df.groupby('cluster')[COLUNAS_CLUSTER + ['n_internacoes']].mean().round(3)
    perfil['n_hospitais'] = df.groupby('cluster').size()
    return perfil.reset_index()


if __name__ == '__main__':
    from src.data.load_data import pipeline_completo
    from src.features.engenharia_features import calcular_ocupacao_diaria, calcular_indicadores_hospital

    internacoes, hospitais = pipeline_completo()
    ocupacao = calcular_ocupacao_diaria(internacoes, hospitais)
    indicadores = calcular_indicadores_hospital(
        internacoes, ocupacao, hospitais, data_limite=internacoes['data_internacao'].max()
    )

    X_escalado, _ = preparar_dados_cluster(indicadores)
    tabela_k = escolher_k(X_escalado)
    logger.info(f'\n{tabela_k}')

    melhor_k = int(tabela_k.loc[tabela_k['silhouette'].idxmax(), 'k'])
    modelo = treinar_kmeans(X_escalado, melhor_k)
    perfil = descrever_clusters(indicadores, modelo.labels_)
    logger.info(f'\n{perfil}')
