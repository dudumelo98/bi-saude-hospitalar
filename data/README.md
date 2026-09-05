# Documentação de Dados

Este diretório segue a mesma divisão em camadas que uso nos outros projetos do portfólio.

## Estrutura de Pastas

- **`raw/`**: Dados originais, exatamente como chegam da fonte (ou do gerador simulado). Nunca edito arquivos aqui na mão.
- **`processed/`**: Dados depois da limpeza e das transformações feitas em `src/data/load_data.py` — o que os notebooks e modelos realmente consomem.
- **`external/`**: Reservado para fontes complementares, como os indicadores da WHO por país.

---

## Sobre o dataset simulado

Os microdados completos do SIH (Sistema de Internações Hospitalares, DATASUS) não vêm em CSV para download direto — eles ficam disponíveis em arquivos `.dbc` num servidor FTP, e abrir esse formato exige uma biblioteca de descompressão específica (`pyreaddbc`), que nem sempre instala de forma simples em todo ambiente. Para manter este projeto reproduzível em qualquer máquina sem depender desse passo, escrevi um gerador (`src/data/gerar_dataset_simulado.py`) que cria internações com a mesma estrutura de campos do SIH e relações estatísticas plausíveis entre idade, especialidade, tempo de permanência, custo e risco de readmissão.

Isso deixa claro desde já: os números deste projeto — taxas de ocupação, readmissão, mortalidade — são de um dataset sintético, não de hospitais reais. Uso esse dataset para validar o pipeline inteiro de ponta a ponta. A seção seguinte explica como substituir pelos dados reais quando eu (ou quem for reproduzir o projeto) tiver acesso a eles.

### Como o gerador funciona, resumido

- Sorteia especialidade e diagnóstico condicionados à idade do paciente — não faz sentido gerar "obstetrícia" para um paciente de 5 anos.
- Aplica sazonalidade: doenças respiratórias concentram no inverno, causas externas (traumas) sobem um pouco no verão.
- Simula reinternações reais: uma fração dos pacientes recebe mais de uma internação, com o intervalo entre elas ligado a um escore de fragilidade. A readmissão em 30 dias nasce dessa sequência de datas — não é um rótulo sorteado direto, o que evita vazamento de informação no modelo de risco.
- Calibra o número de leitos de cada hospital com a Lei de Little (demanda média = taxa de chegada × tempo médio de permanência), usando uma taxa de ocupação alvo mais alta para hospitais públicos — reflexo direto do problema de superlotação que motiva o projeto.

### Arquivos gerados

**`internacoes_simuladas.csv`**

| Coluna | Descrição |
|---|---|
| `internacao_id` | Identificador único da internação |
| `paciente_id` | Identificador único do paciente (permite rastrear reinternações) |
| `hospital_id` | Hospital onde ocorreu a internação |
| `uf` | Estado do paciente |
| `idade` | Idade em anos |
| `sexo` | F ou M |
| `especialidade` | Especialidade responsável pela internação |
| `cid_capitulo` | Categoria ampla de diagnóstico (equivalente a um capítulo do CID-10) |
| `carater_internacao` | Urgência ou Eletiva |
| `data_internacao` / `data_saida` | Datas de entrada e alta |
| `dias_permanencia` | Tempo de permanência em dias |
| `valor_total` | Valor da internação (equivalente ao valor da AIH no SUS) |
| `obito` | 1 se o paciente morreu durante a internação |
| `readmissao_30d` | 1 se o mesmo paciente teve uma nova internação em até 30 dias após esta alta |

**`hospitais_simulados.csv`**

| Coluna | Descrição |
|---|---|
| `hospital_id` | Identificador único do hospital |
| `uf` | Estado |
| `municipio` | Município (fictício) |
| `tipo_gestao` | Público, Filantrópico ou Privado |
| `leitos_totais` | Capacidade total de leitos |

---

## Como usar os dados reais (SIH e WHO)

### 1. DATASUS — SIH (Sistema de Internações Hospitalares)

Dados reais de internações do SUS, por estado, município, procedimento e período.

- **Portal**: [datasus.saude.gov.br](https://datasus.saude.gov.br) — acesso via FTP (`ftp.datasus.gov.br/dissemin/publicos/SIHSUS/`) ou pelo TabNet para consultas agregadas.
- **Formato**: arquivos `.dbc` (um por UF/mês). Para ler no Python, é necessário instalar `pyreaddbc` ou converter antes com o programa TabWin (Windows).
- **Como plugar no projeto**: depois de converter os `.dbc` para CSV/parquet com as colunas originais do SIH-RD, ajusto `carregar_internacoes()` em `src/data/load_data.py` para renomear as colunas do layout oficial (`IDADE`, `ESPEC`, `DIAG_PRINC`, `DT_INTER`, `DT_SAIDA`, `VAL_TOT`, `MORTE`, entre outras) para o mesmo esquema em português usado neste projeto. O resto do pipeline não precisa mudar.

### 2. WHO — Global Health Observatory (Health Systems Data)

Indicadores hospitalares agregados por país, úteis para comparar a rede simulada com benchmarks internacionais (ocupação média, leitos por 1.000 habitantes, etc.).

- **Portal**: [who.int/data/gho](https://www.who.int/data/gho)
- **Formato**: CSV/Excel para download direto, sem necessidade de autenticação.
- **Localização**: colocar em `data/external/`, já que serve como referência complementar e não como base para o treinamento dos modelos.

---
> [!NOTE]
> Por limite de tamanho do GitHub, os arquivos de dados brutos e processados estão no `.gitignore` e não são commitados. Para reproduzir o projeto do zero, basta rodar `python -m src.data.gerar_dataset_simulado` a partir da raiz do projeto.
