"""
Testes de sanidade do pipeline de dados e features.

Não testo os modelos de ML aqui — isso exigiria fixar seeds e tolerâncias
de forma frágil demais para o benefício que traria. Foco nas regras de
negócio que, se quebrarem silenciosamente, contaminam todo o resto do
pipeline sem dar nenhum erro visível: limpeza, ocupação de leitos e a
exclusão de óbitos da base de readmissão.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import pytest

from src.data.load_data import limpar_internacoes
from src.features.engenharia_features import calcular_ocupacao_diaria, preparar_features_readmissao


@pytest.fixture
def hospitais_exemplo():
    return pd.DataFrame({
        'hospital_id': ['H000', 'H001'],
        'uf': ['SP', 'RJ'],
        'municipio': ['M1-SP', 'M1-RJ'],
        'tipo_gestao': ['Público', 'Privado'],
        'leitos_totais': [10, 5],
    })


@pytest.fixture
def internacoes_exemplo():
    return pd.DataFrame({
        'internacao_id': ['I1', 'I2', 'I3', 'I4'],
        'paciente_id': ['P1', 'P2', 'P3', 'P4'],
        'hospital_id': ['H000', 'H000', 'H001', 'H999'],
        'uf': ['SP', 'SP', 'RJ', 'SP'],
        'idade': [40, 70, 200, 30],
        'sexo': ['F', 'M', 'F', 'M'],
        'especialidade': ['Clínica Médica'] * 4,
        'cid_capitulo': ['Doenças do aparelho circulatório'] * 4,
        'carater_internacao': ['Urgência'] * 4,
        'data_internacao': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-01']),
        'data_saida': pd.to_datetime(['2024-01-03', '2024-01-08', '2024-01-05', '2024-01-04']),
        'dias_permanencia': [2, 3, -5, 3],
        'valor_total': [1000.0, 2000.0, 1500.0, -50.0],
        'obito': [0, 1, 0, 0],
        'readmissao_30d': [0, 0, 0, 0],
    })


def test_limpar_internacoes_remove_registros_invalidos(internacoes_exemplo, hospitais_exemplo):
    resultado = limpar_internacoes(internacoes_exemplo, hospitais_exemplo)

    # I3 tem idade > 110 e permanência negativa, I4 tem valor negativo e
    # hospital_id inexistente no cadastro — as duas devem sair.
    assert set(resultado['internacao_id']) == {'I1', 'I2'}


def test_calcular_ocupacao_diaria_nao_gera_leitos_negativos(internacoes_exemplo, hospitais_exemplo):
    limpo = limpar_internacoes(internacoes_exemplo, hospitais_exemplo)
    ocupacao = calcular_ocupacao_diaria(limpo, hospitais_exemplo)

    assert (ocupacao['leitos_ocupados'] >= 0).all()


def test_preparar_features_readmissao_exclui_obitos(internacoes_exemplo, hospitais_exemplo):
    limpo = limpar_internacoes(internacoes_exemplo, hospitais_exemplo)
    features = preparar_features_readmissao(limpo)

    # I2 teve óbito e não deveria aparecer na base de risco de readmissão.
    assert 'I2' not in features['internacao_id'].values
    assert 'obito' not in features.columns
