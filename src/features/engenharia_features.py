"""
Engenharia de features a partir das internações limpas.

Separo em três blocos porque cada um alimenta um modelo diferente do
projeto: ocupação diária alimenta a previsão de demanda por leitos,
indicadores por hospital alimentam a clusterização, e a matriz de
features de readmissão alimenta o classificador de risco.
"""

import pandas as pd
import numpy as np


def calcular_ocupacao_diaria(internacoes: pd.DataFrame, hospitais: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula quantos leitos cada hospital tem ocupados em cada dia.

    Uma internação ocupa um leito do dia da entrada até o dia da saída,
    inclusive. Em vez de expandir cada internação em uma linha por dia
    (caro em memória para dezenas de milhares de internações), monto dois
    eventos por internação — um +1 na entrada e um -1 no dia seguinte à
    saída — e uso soma cumulativa por hospital. É a mesma lógica de um
    livro-caixa: o saldo do dia é o acumulado dos lançamentos anteriores.
    """
    entradas = internacoes[['hospital_id', 'data_internacao']].rename(
        columns={'data_internacao': 'data'}
    )
    entradas['delta'] = 1

    saidas = internacoes[['hospital_id', 'data_saida']].copy()
    saidas['data'] = saidas['data_saida'] + pd.Timedelta(days=1)
    saidas = saidas[['hospital_id', 'data']]
    saidas['delta'] = -1

    eventos = pd.concat([entradas, saidas], ignore_index=True)
    eventos = eventos.groupby(['hospital_id', 'data'])['delta'].sum().reset_index()

    todas_as_datas = []
    for hospital_id, grupo in eventos.groupby('hospital_id'):
        data_min, data_max = grupo['data'].min(), grupo['data'].max()
        calendario = pd.DataFrame({'data': pd.date_range(data_min, data_max, freq='D')})
        calendario['hospital_id'] = hospital_id
        todas_as_datas.append(calendario)
    calendario_completo = pd.concat(todas_as_datas, ignore_index=True)

    ocupacao = calendario_completo.merge(eventos, on=['hospital_id', 'data'], how='left')
    ocupacao['delta'] = ocupacao['delta'].fillna(0)
    ocupacao = ocupacao.sort_values(['hospital_id', 'data'])
    ocupacao['leitos_ocupados'] = ocupacao.groupby('hospital_id')['delta'].cumsum()

    ocupacao = ocupacao.merge(hospitais[['hospital_id', 'leitos_totais', 'uf']], on='hospital_id')
    ocupacao['taxa_ocupacao'] = (ocupacao['leitos_ocupados'] / ocupacao['leitos_totais']).clip(0, 1.5)

    altas_por_dia = internacoes.groupby(['hospital_id', 'data_saida']).size().reset_index(name='altas')
    altas_por_dia = altas_por_dia.rename(columns={'data_saida': 'data'})
    ocupacao = ocupacao.merge(altas_por_dia, on=['hospital_id', 'data'], how='left')
    ocupacao['altas'] = ocupacao['altas'].fillna(0)

    return ocupacao.drop(columns='delta').reset_index(drop=True)


def recortar_janela_valida(ocupacao_diaria: pd.DataFrame, data_limite: pd.Timestamp = None,
                             dias_aquecimento: int = 90) -> pd.DataFrame:
    """
    Descarta as duas pontas da ocupação diária que não refletem demanda real,
    só o limite da extração dos dados.

    - No início, ainda não existe nenhum paciente internado antes do
      primeiro dia do dataset, então a ocupação começa artificialmente
      baixa e sobe aos poucos até o sistema atingir o regime permanente.
      `dias_aquecimento` descarta esse período de aquecimento (uso 90
      dias — o teto de permanência do dataset — para garantir que nenhum
      paciente "invisível" de antes da extração ainda deveria estar
      internado).
    - No fim, `data_limite` corta a série na última data de internação
      registrada, antes da queda artificial causada pelos últimos
      pacientes recebendo alta sem ninguém novo entrando depois disso.

    Uso essa mesma função tanto para a série da rede quanto para os
    indicadores por hospital, porque os dois herdam o problema do mesmo
    jeito — nenhum dos dois deveria ser calculado sobre o período de
    aquecimento ou sobre a cauda de esvaziamento.
    """
    data_inicio_valida = ocupacao_diaria['data'].min() + pd.Timedelta(days=dias_aquecimento)
    recorte = ocupacao_diaria[ocupacao_diaria['data'] >= data_inicio_valida]

    if data_limite is not None:
        recorte = recorte[recorte['data'] <= data_limite]

    return recorte.reset_index(drop=True)


def calcular_serie_rede(ocupacao_diaria: pd.DataFrame, data_limite: pd.Timestamp = None,
                          dias_aquecimento: int = 90) -> pd.DataFrame:
    """
    Agrega a ocupação diária de todos os hospitais em uma série única da rede.

    Essa é a granularidade que uso para a previsão de demanda — modelar
    hospital por hospital individualmente exigiria muito mais histórico do
    que qualquer unidade isolada tem disponível. Ver recortar_janela_valida()
    para o motivo do corte nas duas pontas da série.
    """
    ocupacao_valida = recortar_janela_valida(ocupacao_diaria, data_limite, dias_aquecimento)

    serie = ocupacao_valida.groupby('data').agg(
        leitos_ocupados=('leitos_ocupados', 'sum'),
        leitos_totais=('leitos_totais', 'sum'),
        altas=('altas', 'sum'),
    ).reset_index()
    serie['taxa_ocupacao_rede'] = serie['leitos_ocupados'] / serie['leitos_totais']

    return serie


def calcular_indicadores_hospital(internacoes: pd.DataFrame, ocupacao_diaria: pd.DataFrame,
                                    hospitais: pd.DataFrame, data_limite: pd.Timestamp = None) -> pd.DataFrame:
    """
    Consolida um indicador por hospital, base para a clusterização de perfis.

    Junto os cinco indicadores citados no escopo do projeto: tempo médio de
    permanência, taxa de readmissão, taxa de ocupação, rotatividade de
    leitos e taxa de mortalidade (proxy operacional ligado a tempo de
    espera e gravidade do caso).
    """
    ocupacao_valida = recortar_janela_valida(ocupacao_diaria, data_limite)

    agregado = internacoes.groupby('hospital_id').agg(
        tempo_medio_permanencia=('dias_permanencia', 'mean'),
        taxa_readmissao=('readmissao_30d', 'mean'),
        taxa_mortalidade=('obito', 'mean'),
        valor_medio_internacao=('valor_total', 'mean'),
        n_internacoes=('internacao_id', 'count'),
    ).reset_index()

    ocupacao_media = ocupacao_valida.groupby('hospital_id').agg(
        taxa_ocupacao_media=('taxa_ocupacao', 'mean'),
    ).reset_index()

    n_dias_periodo = ocupacao_valida['data'].nunique()
    rotatividade = internacoes.groupby('hospital_id').size().reset_index(name='total_altas')
    rotatividade = rotatividade.merge(hospitais[['hospital_id', 'leitos_totais']], on='hospital_id')
    rotatividade['rotatividade_leitos'] = (
        rotatividade['total_altas'] / rotatividade['leitos_totais'] / (n_dias_periodo / 30)
    )

    indicadores = (
        agregado
        .merge(ocupacao_media, on='hospital_id')
        .merge(rotatividade[['hospital_id', 'rotatividade_leitos']], on='hospital_id')
        .merge(hospitais, on='hospital_id')
    )
    return indicadores


def preparar_features_readmissao(internacoes: pd.DataFrame) -> pd.DataFrame:
    """
    Monta a matriz de features usada no classificador de risco de readmissão.

    n_internacoes_anteriores conta só o histórico até a internação atual,
    nunca internações futuras — se eu contasse o total de internações do
    paciente sem esse cuidado, o modelo estaria olhando o futuro para
    prever o próprio futuro.
    """
    df = internacoes.sort_values(['paciente_id', 'data_internacao']).copy()
    df['n_internacoes_anteriores'] = df.groupby('paciente_id').cumcount()

    features = df[[
        'internacao_id', 'idade', 'sexo', 'especialidade', 'cid_capitulo',
        'carater_internacao', 'dias_permanencia', 'valor_total',
        'n_internacoes_anteriores', 'obito', 'readmissao_30d'
    ]].copy()

    # Faz sentido treinar o risco de readmissão apenas em quem recebeu alta
    # viva — paciente que morre na internação não pode ser readmitido.
    features = features[features['obito'] == 0].drop(columns='obito')

    features = pd.get_dummies(
        features, columns=['sexo', 'especialidade', 'cid_capitulo', 'carater_internacao'],
        drop_first=True
    )

    return features.reset_index(drop=True)
