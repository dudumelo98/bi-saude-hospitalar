"""
Funções de visualização reaproveitadas entre os notebooks.

Padronizo aqui o estilo dos gráficos para não repetir a mesma configuração
de figsize, paleta e salvamento em cada notebook.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

FIGURES_PATH = Path(__file__).parents[2] / 'reports' / 'figures'
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')


def configurar_estilo():
    """Aplica a configuração padrão de estilo — chamo isso no topo de cada notebook."""
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.rcParams['font.size'] = 11


def salvar_figura(nome_arquivo: str, dpi: int = 150):
    """Salva a figura atual em reports/figures/ com o layout ajustado."""
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / nome_arquivo, dpi=dpi)


def plot_serie_ocupacao(serie_rede: pd.DataFrame, coluna: str = 'taxa_ocupacao_rede', titulo: str = None):
    """Plota a série de ocupação da rede ao longo do tempo."""
    fig, ax = plt.subplots()
    ax.plot(serie_rede['data'], serie_rede[coluna], linewidth=1.2, color='steelblue')
    ax.set_title(titulo or 'Taxa de ocupação da rede ao longo do tempo')
    ax.set_xlabel('Data')
    ax.set_ylabel('Taxa de ocupação')
    return fig, ax


def plot_matriz_confusao(matriz, labels=('Não readmite', 'Readmite')):
    """Plota a matriz de confusão do classificador de risco de readmissão."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de confusão — risco de readmissão')
    return fig, ax


def plot_importancia_features(importancia: pd.DataFrame, top_n: int = 10):
    """Plota as features mais relevantes do classificador."""
    top = importancia.head(top_n).sort_values('importancia')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top['feature'], top['importancia'], color='steelblue')
    ax.set_xlabel('Importância')
    ax.set_title(f'Top {top_n} features — risco de readmissão')
    return fig, ax
