"""
Script de ingestão e limpeza das internações hospitalares.

Centralizo aqui a lógica de carregamento para que os notebooks não
precisem repetir as mesmas transformações. Se um dia eu trocar o dataset
simulado pelos microdados reais do SIH, só preciso ajustar a função
carregar_internacoes() — o resto do pipeline espera as mesmas colunas.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_PATH = Path(__file__).parents[2] / 'data' / 'raw'
PROCESSED_PATH = Path(__file__).parents[2] / 'data' / 'processed'


def carregar_internacoes(caminho: Path = None) -> pd.DataFrame:
    """Carrega o arquivo de internações, sem nenhuma limpeza aplicada ainda."""
    if caminho is None:
        caminho = RAW_PATH / 'internacoes_simuladas.csv'

    logger.info(f'Carregando internações de {caminho}')
    df = pd.read_csv(caminho, parse_dates=['data_internacao', 'data_saida'])
    logger.info(f'Internações carregadas: {len(df):,} linhas, {df.shape[1]} colunas')
    return df


def carregar_hospitais(caminho: Path = None) -> pd.DataFrame:
    """Carrega o cadastro de hospitais (capacidade de leitos, UF, tipo de gestão)."""
    if caminho is None:
        caminho = RAW_PATH / 'hospitais_simulados.csv'

    logger.info(f'Carregando cadastro de hospitais de {caminho}')
    df = pd.read_csv(caminho)
    logger.info(f'Hospitais carregados: {len(df):,} linhas')
    return df


def limpar_internacoes(df: pd.DataFrame, hospitais: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a limpeza definida durante a EDA.

    Decisões tomadas:
    - Removo internações com data_saida anterior à data_internacao — erro de
      digitação em cadastro real, e aqui serve como verificação de sanidade
      do próprio gerador de dados.
    - Removo dias_permanencia fora da faixa 1-90 dias — acima disso é mais
      provável ser erro de registro do que internação real.
    - Removo idade fora da faixa 0-110 anos.
    - Removo valor_total menor ou igual a zero.
    - Removo internações que referenciam hospital_id inexistente no cadastro.
    - Removo duplicatas exatas de internacao_id.
    """
    n_inicial = len(df)

    df = df[df['data_saida'] >= df['data_internacao']]
    logger.info(f'Após remover datas inconsistentes: {len(df):,} linhas')

    df = df[df['dias_permanencia'].between(1, 90)]
    logger.info(f'Após remover permanência fora de 1-90 dias: {len(df):,} linhas')

    df = df[df['idade'].between(0, 110)]
    logger.info(f'Após remover idade inválida: {len(df):,} linhas')

    df = df[df['valor_total'] > 0]
    logger.info(f'Após remover valor_total inválido: {len(df):,} linhas')

    df = df[df['hospital_id'].isin(hospitais['hospital_id'])]
    logger.info(f'Após remover hospital_id sem cadastro: {len(df):,} linhas')

    df = df.drop_duplicates(subset='internacao_id')
    logger.info(f'Após remover duplicatas: {len(df):,} linhas')

    # Tipos e categorias — deixo explícito como category porque essas
    # colunas entram em groupby com frequência no resto do pipeline.
    for col in ['sexo', 'especialidade', 'cid_capitulo', 'carater_internacao', 'uf']:
        df[col] = df[col].astype('category')
    df['obito'] = df['obito'].astype(int)
    df['readmissao_30d'] = df['readmissao_30d'].astype(int)

    n_final = len(df)
    logger.info(f'Limpeza concluída: {n_inicial - n_final:,} linhas removidas ({(1 - n_final/n_inicial)*100:.2f}%)')

    return df.reset_index(drop=True)


def pipeline_completo() -> tuple:
    """
    Executa o pipeline completo de ingestão e retorna (internações limpas, hospitais).

    Uso como ponto de entrada nos notebooks para evitar repetir a mesma
    sequência de carregamento e limpeza em cada um deles.
    """
    hospitais = carregar_hospitais()
    internacoes_raw = carregar_internacoes()
    internacoes = limpar_internacoes(internacoes_raw, hospitais)

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    internacoes.to_parquet(PROCESSED_PATH / 'internacoes_limpas.parquet', index=False)
    hospitais.to_parquet(PROCESSED_PATH / 'hospitais.parquet', index=False)
    logger.info(f'Dados processados salvos em {PROCESSED_PATH}')

    return internacoes, hospitais


if __name__ == '__main__':
    pipeline_completo()
