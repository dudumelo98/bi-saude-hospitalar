"""
Previsão de demanda por leitos com Prophet.

Escolhi Prophet como modelo principal porque a série de ocupação hospitalar
tem duas sazonalidades relevantes ao mesmo tempo — semanal (internações
eletivas concentram em dias de semana) e anual (pico respiratório no
inverno) — e o Prophet trata as duas de forma nativa, sem eu precisar
diferenciar a série manualmente como seria necessário num ARIMA clássico.

Para benchmark, também deixo aqui uma função com Holt-Winters
(statsmodels), que serve de comparação mais simples e mais rápida de
treinar quando não faz sentido instalar o Prophet inteiro só para validar
uma ideia rápida.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_PATH = Path(__file__).parents[2] / 'models'
MODELS_PATH.mkdir(exist_ok=True)


def preparar_serie_prophet(serie_rede: pd.DataFrame, coluna_alvo: str = 'leitos_ocupados') -> pd.DataFrame:
    """Formata a série da rede no layout exigido pelo Prophet (colunas ds e y)."""
    serie = serie_rede[['data', coluna_alvo]].rename(columns={'data': 'ds', coluna_alvo: 'y'})
    return serie.sort_values('ds').reset_index(drop=True)


def treinar_prophet(serie: pd.DataFrame):
    """
    Treina um Prophet com sazonalidade semanal e anual ativadas.

    Uso changepoint_prior_scale baixo porque a capacidade instalada de
    leitos não muda de uma hora para outra — quero uma tendência suave, não
    um modelo que reage a cada oscilação de curto prazo como se fosse uma
    mudança estrutural.
    """
    from prophet import Prophet

    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.03,
        interval_width=0.90,
    )
    modelo.fit(serie)
    return modelo


def avaliar_modelo(modelo, serie: pd.DataFrame, horizonte_dias: int = 30) -> Dict:
    """
    Avalia o modelo no horizonte de previsão definido, usando os últimos
    `horizonte_dias` da série como conjunto de teste (out-of-sample).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    corte = serie['ds'].max() - pd.Timedelta(days=horizonte_dias)
    treino = serie[serie['ds'] <= corte]
    teste = serie[serie['ds'] > corte]

    if len(teste) == 0:
        logger.warning('Sem dados no período de teste')
        return {}

    futuro = modelo.make_future_dataframe(periods=horizonte_dias)
    previsao = modelo.predict(futuro)

    prev_teste = previsao[previsao['ds'].isin(teste['ds'])][['ds', 'yhat']]
    resultado = teste.merge(prev_teste, on='ds')

    metricas = {
        'mae': mean_absolute_error(resultado['y'], resultado['yhat']),
        'rmse': np.sqrt(mean_squared_error(resultado['y'], resultado['yhat'])),
        'mape': np.mean(np.abs((resultado['y'] - resultado['yhat']) / resultado['y'])) * 100,
        'n_dias_teste': len(resultado),
    }
    return metricas


def prever_ocupacao(serie_rede: pd.DataFrame, horizonte: int = 30, coluna_alvo: str = 'taxa_ocupacao_rede') -> pd.DataFrame:
    """Pipeline completo: prepara a série, treina o Prophet e devolve a previsão futura."""
    serie = preparar_serie_prophet(serie_rede, coluna_alvo=coluna_alvo)
    modelo = treinar_prophet(serie)

    futuro = modelo.make_future_dataframe(periods=horizonte)
    previsao = modelo.predict(futuro)

    previsao_futura = previsao[previsao['ds'] > serie['ds'].max()][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].copy()
    previsao_futura['yhat'] = previsao_futura['yhat'].clip(lower=0)
    previsao_futura['yhat_lower'] = previsao_futura['yhat_lower'].clip(lower=0)

    return previsao_futura


def treinar_baseline_holt_winters(serie: pd.DataFrame, horizonte: int = 30):
    """
    Baseline estatístico com suavização exponencial (Holt-Winters).

    Uso essa função para ter um número de referência rápido antes de subir
    o Prophet completo — se o Holt-Winters já erra pouco, isso me diz que a
    série tem pouca complexidade não-linear e o ganho do Prophet pode ser
    pequeno; se erra muito, é sinal de que vale investir no modelo mais
    sofisticado.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_treino = serie['y'].values
    modelo = ExponentialSmoothing(
        y_treino, trend='add', seasonal='add', seasonal_periods=7,
        initialization_method='estimated'
    ).fit()
    previsao = modelo.forecast(horizonte)
    return modelo, previsao


if __name__ == '__main__':
    from src.data.load_data import pipeline_completo
    from src.features.engenharia_features import calcular_ocupacao_diaria, calcular_serie_rede

    internacoes, hospitais = pipeline_completo()
    ocupacao = calcular_ocupacao_diaria(internacoes, hospitais)
    serie_rede = calcular_serie_rede(ocupacao, data_limite=internacoes['data_internacao'].max())

    previsao = prever_ocupacao(serie_rede)
    logger.info(f'Previsão gerada para os próximos {len(previsao)} dias')
