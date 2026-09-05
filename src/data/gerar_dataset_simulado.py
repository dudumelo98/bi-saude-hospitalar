"""
Gerador do dataset simulado de internações hospitalares.

Os microdados completos do SIH (DATASUS) exigem baixar arquivos .dbc do FTP
oficial e descomprimi-los com uma biblioteca específica (pyreaddbc), o que
não é trivial de reproduzir em qualquer máquina. Para manter o projeto
100% reproduzível sem depender desse download, construí aqui um gerador que
simula internações com a mesma estrutura de campos do SIH-RD e com relações
estatísticas plausíveis entre idade, especialidade, tempo de permanência,
custo e readmissão.

Isso me permite testar o pipeline inteiro de ponta a ponta. Quando os dados
reais do SIH estiverem disponíveis (ver data/README.md para o passo a
passo), basta apontar load_data.py para o arquivo real — o resto do
pipeline não muda, porque as colunas seguem o mesmo layout.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parents[2] / 'data' / 'raw'

SEED = 42

UFS = ['SP', 'RJ', 'MG', 'BA', 'RS', 'PR', 'PE', 'CE', 'DF']

ESPECIALIDADES = [
    'Clínica Médica', 'Cirurgia', 'Obstetrícia', 'Pediatria',
    'Psiquiatria', 'UTI', 'Ortopedia'
]

# Categoria ampla de diagnóstico (equivalente a capítulo do CID-10),
# usada para não trabalhar com os milhares de códigos individuais do CID.
CID_CAPITULOS = [
    'Doenças do aparelho circulatório', 'Doenças do aparelho respiratório',
    'Doenças do aparelho digestivo', 'Neoplasias', 'Causas externas (traumas)',
    'Doenças infecciosas e parasitárias', 'Gravidez, parto e puerpério',
    'Transtornos mentais e comportamentais', 'Outras causas'
]

# Parâmetros de tempo de permanência (dias) por especialidade — média e
# dispersão usadas para amostrar de uma distribuição gama, que só gera
# valores positivos e tem a assimetria à direita que internações costumam ter.
PARAMS_PERMANENCIA = {
    'Clínica Médica': (5.0, 1.6),
    'Cirurgia': (4.0, 1.4),
    'Obstetrícia': (2.5, 1.2),
    'Pediatria': (3.5, 1.5),
    'Psiquiatria': (12.0, 2.0),
    'UTI': (9.0, 1.8),
    'Ortopedia': (4.5, 1.5),
}

# Valor médio da AIH (R$) por especialidade — usado como base do custo,
# depois ajustado pelo tempo de permanência real de cada internação.
VALOR_BASE_ESPECIALIDADE = {
    'Clínica Médica': 1400, 'Cirurgia': 3200, 'Obstetrícia': 900,
    'Pediatria': 1100, 'Psiquiatria': 1800, 'UTI': 5200, 'Ortopedia': 2600,
}


def _gerar_hospitais(n_hospitais: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Gera o cadastro de hospitais participantes da rede simulada.

    Nesta etapa eu só defino UF, tipo de gestão e um peso de porte relativo
    — ainda não defino o número de leitos. Se eu sorteasse leitos_totais de
    forma totalmente independente do volume de internações, a taxa de
    ocupação resultante ficaria artificialmente baixa ou alta dependendo da
    sorte. Calibro o número real de leitos depois, em
    _calibrar_leitos_totais(), com base na demanda que cada hospital de
    fato recebeu na simulação.
    """
    tipo_gestao = rng.choice(
        ['Público', 'Filantrópico', 'Privado'], size=n_hospitais, p=[0.45, 0.30, 0.25]
    )
    uf = rng.choice(UFS, size=n_hospitais)
    peso_porte = rng.lognormal(mean=0, sigma=0.9, size=n_hospitais)

    hospitais = pd.DataFrame({
        'hospital_id': [f'H{i:03d}' for i in range(n_hospitais)],
        'uf': uf,
        'municipio': [f'Município {rng.integers(1, 25)}-{u}' for u in uf],
        'tipo_gestao': tipo_gestao,
        'peso_porte': peso_porte,
    })
    return hospitais


def _calibrar_leitos_totais(df: pd.DataFrame, hospitais: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Define o número de leitos de cada hospital a partir da demanda simulada.

    Uso a Lei de Little (L = λ × W) para estimar quantos pacientes cada
    hospital mantém internados ao mesmo tempo, em média: λ é a taxa de
    chegada de internações por dia e W é o tempo médio de permanência.
    Dividindo essa média pela taxa de ocupação que eu quero para aquele
    hospital, chego no número de leitos que ele precisaria ter.

    Puxo a taxa de ocupação alvo para cima em hospitais públicos — é o
    reflexo direto do problema descrito no escopo do projeto, de que a rede
    pública sofre mais com superlotação do que a rede privada.
    """
    periodo_dias = (df['data_saida'].max() - df['data_internacao'].min()).days

    demanda = df.groupby('hospital_id').agg(
        n_admissoes=('internacao_id', 'count'),
        los_medio=('dias_permanencia', 'mean'),
    ).reset_index()
    demanda['lambda_dia'] = demanda['n_admissoes'] / periodo_dias
    demanda['ocupacao_media_leitos'] = demanda['lambda_dia'] * demanda['los_medio']

    faixa_ocupacao_alvo = {
        'Público': (0.75, 1.08),
        'Filantrópico': (0.65, 0.92),
        'Privado': (0.45, 0.75),
    }
    hospitais = hospitais.copy()
    ocupacao_alvo = np.empty(len(hospitais))
    for tipo, (baixo, alto) in faixa_ocupacao_alvo.items():
        mask = (hospitais['tipo_gestao'] == tipo).values
        ocupacao_alvo[mask] = rng.uniform(baixo, alto, size=mask.sum())
    hospitais['ocupacao_alvo'] = ocupacao_alvo

    hospitais = hospitais.merge(demanda, on='hospital_id', how='left')
    hospitais['ocupacao_media_leitos'] = hospitais['ocupacao_media_leitos'].fillna(1.0)
    hospitais['leitos_totais'] = np.maximum(
        15, np.ceil(hospitais['ocupacao_media_leitos'] / hospitais['ocupacao_alvo'])
    ).astype(int)

    return hospitais.drop(columns=['peso_porte', 'ocupacao_alvo', 'n_admissoes', 'los_medio', 'lambda_dia', 'ocupacao_media_leitos'])


def _sortear_especialidade_e_cid(idade: np.ndarray, rng: np.random.Generator) -> tuple:
    """
    Sorteia especialidade e categoria de CID condicionados à idade.

    Não faz sentido sortear isso de forma independente da idade — pediatria
    em paciente de 80 anos ou obstetrícia em paciente de 5 anos não existem
    na prática, e um modelo treinado nesses dados aprenderia ruído em vez de
    padrão clínico real.
    """
    n = len(idade)
    especialidade = np.empty(n, dtype=object)
    cid = np.empty(n, dtype=object)

    faixa_crianca = idade < 12
    faixa_jovem = (idade >= 12) & (idade < 40)
    faixa_adulta = (idade >= 40) & (idade < 65)
    faixa_idosa = idade >= 65

    especialidade[faixa_crianca] = rng.choice(
        ['Pediatria', 'Cirurgia', 'Ortopedia'], size=faixa_crianca.sum(), p=[0.7, 0.2, 0.1]
    )
    especialidade[faixa_jovem] = rng.choice(
        ESPECIALIDADES, size=faixa_jovem.sum(),
        p=[0.10, 0.20, 0.28, 0.02, 0.10, 0.08, 0.22]
    )
    especialidade[faixa_adulta] = rng.choice(
        ESPECIALIDADES, size=faixa_adulta.sum(),
        p=[0.24, 0.22, 0.03, 0.01, 0.09, 0.16, 0.25]
    )
    especialidade[faixa_idosa] = rng.choice(
        ESPECIALIDADES, size=faixa_idosa.sum(),
        p=[0.34, 0.16, 0.00, 0.01, 0.05, 0.28, 0.16]
    )

    for spec, mapa_cid in {
        'Clínica Médica': ['Doenças do aparelho circulatório', 'Doenças do aparelho respiratório',
                            'Doenças do aparelho digestivo', 'Doenças infecciosas e parasitárias'],
        'Cirurgia': ['Doenças do aparelho digestivo', 'Neoplasias', 'Causas externas (traumas)'],
        'Obstetrícia': ['Gravidez, parto e puerpério'],
        'Pediatria': ['Doenças do aparelho respiratório', 'Doenças infecciosas e parasitárias',
                       'Outras causas'],
        'Psiquiatria': ['Transtornos mentais e comportamentais'],
        'UTI': ['Doenças do aparelho circulatório', 'Doenças do aparelho respiratório',
                'Causas externas (traumas)'],
        'Ortopedia': ['Causas externas (traumas)', 'Outras causas'],
    }.items():
        mask = especialidade == spec
        cid[mask] = rng.choice(mapa_cid, size=mask.sum())

    return especialidade, cid


def _sortear_mes_com_sazonalidade(especialidade: np.ndarray, n_dias: int, data_inicio: pd.Timestamp,
                                    rng: np.random.Generator) -> np.ndarray:
    """
    Distribui as internações no tempo com sazonalidade dependente da especialidade.

    Doença respiratória concentra no inverno (junho a agosto no Brasil).
    Causas externas sobem um pouco no verão e em dezembro/janeiro, por causa
    de festas de fim de ano e maior circulação nas ruas. As demais
    especialidades ficam com distribuição praticamente uniforme ao longo do ano.
    """
    dias_base = rng.integers(0, n_dias, size=len(especialidade))
    datas = data_inicio + pd.to_timedelta(dias_base, unit='D')
    meses = datas.month.values

    peso_respiratorio = np.where(np.isin(meses, [6, 7, 8]), 2.2, 1.0)
    peso_trauma = np.where(np.isin(meses, [12, 1, 2]), 1.6, 1.0)

    aceitar = rng.random(len(especialidade))
    reforcar_resp = (especialidade == 'Clínica Médica') & (aceitar < (peso_respiratorio - 1) / 2.2)
    reforcar_trauma = (np.isin(especialidade, ['Ortopedia', 'UTI'])) & (aceitar < (peso_trauma - 1) / 1.6)

    # Para os casos "reforçados", puxo a data para dentro da janela de pico
    # em vez de descartar o registro — assim não perco volume total de linhas.
    n_reforco = reforcar_resp.sum()
    if n_reforco > 0:
        anos_disponiveis = sorted(set(data_inicio.year + np.arange((n_dias // 365) + 1)))
        ano_sorteado = rng.choice(anos_disponiveis, size=n_reforco)
        mes_sorteado = rng.choice([6, 7, 8], size=n_reforco)
        dia_sorteado = rng.integers(1, 28, size=n_reforco)
        novas_datas = pd.to_datetime({'year': ano_sorteado, 'month': mes_sorteado, 'day': dia_sorteado})
        datas.values[reforcar_resp] = novas_datas.values

    return datas


def gerar_dataset(
    n_hospitais: int = 16,
    n_pacientes: int = 52000,
    n_internacoes: int = 85000,
    data_inicio: str = '2022-01-01',
    n_dias: int = 1095,
    seed: int = SEED,
) -> tuple:
    """
    Gera o par (internações, hospitais) usado no resto do projeto.

    A readmissão em 30 dias não é sorteada diretamente — ela nasce da
    sequência real de datas de cada paciente. Uma parte dos pacientes recebe
    mais de uma internação, com o intervalo entre elas dependendo de um
    escore de fragilidade ligado à idade. Isso evita vazamento de
    informação: o modelo de risco vai precisar aprender o padrão a partir de
    idade, especialidade e tempo de permanência, não de um rótulo copiado.
    """
    rng = np.random.default_rng(seed)
    data_inicio_ts = pd.Timestamp(data_inicio)

    hospitais = _gerar_hospitais(n_hospitais, rng)

    # Escore de fragilidade por paciente: dirige tanto a idade quanto a
    # chance de reinternação. Pacientes mais fragilizados tendem a ser mais
    # velhos e a voltar ao hospital com mais frequência.
    fragilidade = rng.beta(2, 5, size=n_pacientes)
    idade_paciente = np.clip(rng.normal(35 + fragilidade * 55, 14), 0, 98).astype(int)
    uf_paciente = rng.choice(UFS, size=n_pacientes)

    # Número de internações por paciente: a maioria tem só uma, mas quem tem
    # fragilidade alta tende a acumular mais passagens pelo hospital.
    lambda_internacoes = 1 + fragilidade * 3.5
    n_por_paciente = 1 + rng.poisson(lambda_internacoes - 1)
    n_por_paciente = np.clip(n_por_paciente, 1, 6)

    # Ajusto o total para bater aproximadamente com n_internacoes pedido.
    fator_ajuste = n_internacoes / n_por_paciente.sum()
    if fator_ajuste < 1:
        indices_mantidos = rng.choice(n_pacientes, size=int(n_pacientes * fator_ajuste), replace=False)
        mask = np.zeros(n_pacientes, dtype=bool)
        mask[indices_mantidos] = True
        idade_paciente = idade_paciente[mask]
        uf_paciente = uf_paciente[mask]
        fragilidade = fragilidade[mask]
        n_por_paciente = n_por_paciente[mask]

    paciente_id = np.repeat(np.arange(len(idade_paciente)), n_por_paciente)
    idade = np.repeat(idade_paciente, n_por_paciente)
    uf = np.repeat(uf_paciente, n_por_paciente)
    fragilidade_rep = np.repeat(fragilidade, n_por_paciente)
    ordem_internacao = np.concatenate([np.arange(k) for k in n_por_paciente])

    n_total = len(paciente_id)
    sexo = rng.choice(['F', 'M'], size=n_total, p=[0.54, 0.46])

    especialidade, cid_capitulo = _sortear_especialidade_e_cid(idade, rng)

    # Tempo de permanência: gama por especialidade, com um acréscimo suave
    # para pacientes mais velhos (recuperação mais lenta é um padrão clínico
    # bem documentado).
    dias_permanencia = np.empty(n_total)
    for spec, (media, forma) in PARAMS_PERMANENCIA.items():
        mask = especialidade == spec
        escala = media / forma
        base = rng.gamma(shape=forma, scale=escala, size=mask.sum())
        ajuste_idade = 1 + np.clip((idade[mask] - 60), 0, None) * 0.01
        dias_permanencia[mask] = np.clip(base * ajuste_idade, 1, 60)
    dias_permanencia = np.round(dias_permanencia).astype(int)

    # Caráter da internação: urgência domina causas externas e circulatório,
    # eletiva é mais comum em cirurgia programada.
    carater = np.where(
        np.isin(cid_capitulo, ['Causas externas (traumas)', 'Doenças do aparelho circulatório']),
        rng.choice(['Urgência', 'Eletiva'], size=n_total, p=[0.88, 0.12]),
        rng.choice(['Urgência', 'Eletiva'], size=n_total, p=[0.55, 0.45])
    )

    # Custo (AIH): base por especialidade, cresce com o tempo de permanência
    # e recebe ruído multiplicativo para não ficar determinístico demais.
    valor_base = np.array([VALOR_BASE_ESPECIALIDADE[e] for e in especialidade])
    ruido = rng.lognormal(mean=0, sigma=0.25, size=n_total)
    valor_total = np.round(valor_base * (1 + dias_permanencia * 0.12) * ruido, 2)

    # Óbito: probabilidade logística em função de idade, especialidade de
    # UTI e permanência muito longa — os três fatores de risco clínico mais
    # citados na literatura de mortalidade hospitalar.
    logit = (
        -7.6
        + 0.05 * idade
        + 1.4 * (especialidade == 'UTI')
        + 0.03 * dias_permanencia
        + 0.6 * (carater == 'Urgência')
    )
    prob_obito = 1 / (1 + np.exp(-logit))
    obito = (rng.random(n_total) < prob_obito).astype(int)

    # Datas: primeiro sorteio com sazonalidade, depois força a ordem
    # cronológica das internações de um mesmo paciente e faz o intervalo
    # entre internações depender da fragilidade — é isso que cria o sinal de
    # readmissão sem precisar rotular manualmente.
    data_internacao = _sortear_mes_com_sazonalidade(especialidade, n_dias, data_inicio_ts, rng)
    data_internacao = pd.Series(data_internacao)

    df = pd.DataFrame({
        'paciente_id': paciente_id,
        'ordem_internacao': ordem_internacao,
        'fragilidade': fragilidade_rep,
        'uf': uf,
        'idade': idade,
        'sexo': sexo,
        'especialidade': especialidade,
        'cid_capitulo': cid_capitulo,
        'dias_permanencia': dias_permanencia,
        'carater_internacao': carater,
        'valor_total': valor_total,
        'obito': obito,
        'data_internacao': data_internacao,
    })

    # Reconstruo a linha do tempo de cada paciente: a primeira internação
    # usa a data sorteada, as seguintes saem da alta da anterior mais um
    # intervalo que fica menor quanto maior a fragilidade do paciente.
    df = df.sort_values(['paciente_id', 'ordem_internacao']).reset_index(drop=True)
    datas_finais = df['data_internacao'].copy()
    for idx in df.index[1:]:
        if df.loc[idx, 'ordem_internacao'] == 0:
            continue
        idx_anterior = idx - 1
        alta_anterior = datas_finais[idx_anterior] + pd.Timedelta(days=int(df.loc[idx_anterior, 'dias_permanencia']))
        intervalo_medio = 90 - df.loc[idx, 'fragilidade'] * 75  # de ~15 a ~90 dias
        intervalo = max(1, int(rng.exponential(intervalo_medio)))
        datas_finais[idx] = alta_anterior + pd.Timedelta(days=intervalo)
    df['data_internacao'] = datas_finais
    df['data_saida'] = df['data_internacao'] + pd.to_timedelta(df['dias_permanencia'], unit='D')

    # Hospital: sorteado dentro do mesmo estado do paciente, com viés para
    # hospitais de maior porte — replica o padrão real de regionalização do
    # SUS, em que unidades maiores absorvem proporcionalmente mais casos.
    hospitais_por_uf = {u: hospitais[hospitais['uf'] == u] for u in UFS}
    hospital_id = np.empty(n_total, dtype=object)
    for u in UFS:
        mask = df['uf'] == u
        candidatos = hospitais_por_uf[u]
        if candidatos.empty:
            candidatos = hospitais
        pesos = candidatos['peso_porte'].values / candidatos['peso_porte'].sum()
        hospital_id[mask.values] = rng.choice(candidatos['hospital_id'].values, size=mask.sum(), p=pesos)
    df['hospital_id'] = hospital_id

    # Readmissão em 30 dias: verdadeiro quando a próxima internação do mesmo
    # paciente começa em até 30 dias após a alta da internação atual.
    df = df.sort_values(['paciente_id', 'data_internacao']).reset_index(drop=True)
    proxima_internacao = df.groupby('paciente_id')['data_internacao'].shift(-1)
    dias_para_proxima = (proxima_internacao - df['data_saida']).dt.days
    df['readmissao_30d'] = ((dias_para_proxima >= 0) & (dias_para_proxima <= 30)).astype(int)

    # As reinternações de pacientes fragilizados vão empurrando a data para
    # frente na cadeia simulada, e algumas acabam caindo além da janela de
    # observação pedida. Eu já usei essas ocorrências futuras para marcar a
    # readmissão da internação anterior — agora descarto as linhas que
    # caíram fora da janela, para manter o dataset final com um período
    # limpo e bem definido, como aconteceria numa extração real de dados.
    data_fim_ts = data_inicio_ts + pd.Timedelta(days=n_dias - 1)
    df = df[(df['data_internacao'] >= data_inicio_ts) & (df['data_internacao'] <= data_fim_ts)].reset_index(drop=True)

    df['internacao_id'] = [f'I{i:07d}' for i in range(len(df))]
    df['paciente_id'] = 'P' + df['paciente_id'].astype(str).str.zfill(6)

    colunas_finais = [
        'internacao_id', 'paciente_id', 'hospital_id', 'uf', 'idade', 'sexo',
        'especialidade', 'cid_capitulo', 'carater_internacao', 'data_internacao',
        'data_saida', 'dias_permanencia', 'valor_total', 'obito', 'readmissao_30d'
    ]
    df = df[colunas_finais].sort_values('data_internacao').reset_index(drop=True)

    hospitais = _calibrar_leitos_totais(df, hospitais, rng)

    return df, hospitais


def salvar_dataset(df: pd.DataFrame, hospitais: pd.DataFrame) -> None:
    """Salva os dois arquivos em data/raw/, prontos para o load_data.py ler."""
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH / 'internacoes_simuladas.csv', index=False)
    hospitais.to_csv(RAW_PATH / 'hospitais_simulados.csv', index=False)
    print(f'{len(df):,} internações salvas em {RAW_PATH / "internacoes_simuladas.csv"}')
    print(f'{len(hospitais):,} hospitais salvos em {RAW_PATH / "hospitais_simulados.csv"}')


if __name__ == '__main__':
    internacoes, hospitais = gerar_dataset()
    salvar_dataset(internacoes, hospitais)
