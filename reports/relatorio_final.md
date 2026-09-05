# Relatório Final ,BI de Saúde Hospitalar e Otimização de Leitos

## Resumo Executivo

Este projeto constrói um pipeline de inteligência hospitalar para três problemas ligados entre si: prever a demanda por leitos, identificar pacientes com maior risco de readmissão em 30 dias e agrupar hospitais por perfil operacional para priorizar onde intervir primeiro.

Como os microdados completos do SIH (DATASUS) exigem baixar e descomprimir arquivos `.dbc` de um servidor FTP — um passo que não é trivial de reproduzir em qualquer máquina —, construí um gerador que simula uma rede de 16 hospitais e 80.139 internações ao longo de três anos (2022 a 2024), com a mesma estrutura de campos do SIH-RD e relações estatísticas plausíveis entre idade, especialidade, permanência, custo e risco. O `data/README.md` documenta exatamente como plugar os dados reais do SIH e da WHO no mesmo pipeline quando estiverem disponíveis.

A rede simulada opera com ocupação média de 91% e picos que passam de 100% em vários hospitais — o cenário de superlotação que motiva o projeto. Um baseline de suavização exponencial já prevê a ocupação da rede com erro médio de 6% no horizonte de 30 dias. O classificador de risco de readmissão alcançou AUC de 0,60, moderado mas coerente com o que a literatura reporta para modelos que usam só dados administrativos. A clusterização separou os hospitais em 5 perfis operacionais, do "equilibrado" ao "crise aguda de leitos".

## Descrição do Dataset

O dataset simulado tem duas tabelas: 80.139 internações e o cadastro de 16 hospitais distribuídos em 9 estados, com gestão pública, filantrópica e privada. Cada internação registra idade, sexo, especialidade, categoria de diagnóstico, caráter (urgência ou eletiva), datas de entrada e saída, valor da internação, óbito e se houve uma nova internação do mesmo paciente em até 30 dias.

Principais números da base, depois da limpeza:

| Indicador | Valor |
|---|---|
| Período coberto | 01/01/2022 a 30/12/2024 |
| Total de internações | 80.139 |
| Hospitais na rede | 16 |
| Tempo médio de permanência | 6,0 dias |
| Taxa de mortalidade geral | 2,9% |
| Taxa de readmissão em 30 dias | 20,4% |
| Taxa de ocupação média da rede | 91,2% |

Não foram encontradas duplicatas nem datas inconsistentes na EDA — esperado, já que o dataset nasce simulado. Ainda assim, o pipeline aplica todas as validações de sanidade (permanência entre 1 e 90 dias, idade entre 0 e 110 anos, valor positivo, hospital cadastrado) porque é isso que garante que ele não quebra silenciosamente no dia em que eu trocar pelo SIH real.

## Objetivo da Análise

Três perguntas guiaram o trabalho, na mesma ordem dos indicadores citados no escopo do projeto:

1. Como vai se comportar a demanda por leitos nos próximos dias, e quando a rede deve se preparar para picos de ocupação?
2. Quais pacientes, no momento da alta, têm maior risco de voltar a ser internados em até 30 dias?
3. Existem grupos de hospitais com perfil operacional parecido, que permitam priorizar onde uma intervenção (reforço de leitos, revisão de altas, programa de acompanhamento) tem mais impacto?

## Metodologia

A análise seguiu quatro notebooks em sequência, cada um alimentando o próximo.

**EDA (01)** confirmou sazonalidade real nas internações respiratórias concentrada no meio do ano, cauda longa nas distribuições de permanência e valor, e correlação fraca (abaixo de 0,15) entre as variáveis numéricas isoladas e os dois desfechos de interesse — óbito e readmissão.

**Limpeza e features (02)** aplicou as validações de sanidade e calculou a ocupação diária de leitos por hospital através de uma soma cumulativa de eventos de entrada e saída — abordagem que evita expandir cada internação em uma linha por dia de permanência. A série de ocupação teve as duas pontas cortadas: o início, porque o sistema simulado começa sem nenhum paciente acumulado (efeito de aquecimento), e o fim, porque os últimos pacientes internados recebem alta sem que ninguém novo entre depois do corte da extração. Nenhum dos dois efeitos é demanda real.

**Previsão de ocupação (03)** comparou um baseline de suavização exponencial (Holt-Winters) com Prophet como modelo de produção recomendado. Prophet foi escolhido pela literatura de forecasting por tratar sazonalidade semanal e anual de forma simultânea e nativa — relevante aqui porque a série tem as duas ao mesmo tempo.

**Risco de readmissão (04)** comparou regressão logística (baseline interpretável) com Random Forest, treinadas sobre idade, especialidade, diagnóstico, caráter da internação, permanência, valor e histórico de internações anteriores do paciente — excluindo quem morreu durante a internação, já que não pode ser readmitido.

**Clusterização de hospitais (05)** padronizou cinco indicadores por hospital (permanência, readmissão, mortalidade, ocupação, rotatividade de leitos) e aplicou K-Means, escolhendo o número de grupos pelo silhouette score.

## Principais Achados

**A rede opera no limite, não em crise uniforme.** A ocupação média de 91% já é alta para qualquer padrão hospitalar, mas a clusterização mostra que o problema não está distribuído igualmente: existe um grupo pequeno de hospitais com ocupação acima de 115% e a maior taxa de readmissão da rede — esse é o grupo que concentra o problema descrito no escopo do projeto, e onde uma intervenção tem o maior retorno por esforço.

**A previsão de ocupação de curto prazo é mais fácil do que parece.** O baseline Holt-Winters, que é praticamente o modelo mais simples possível para uma série sazonal, já entrega MAPE de 6,1% no horizonte de 30 dias. Isso sugere que grande parte da variação da ocupação é sazonal e regular — o ganho de um modelo mais sofisticado como o Prophet deve aparecer mais em horizontes mais longos (60-90 dias), quando a sazonalidade anual pesa proporcionalmente mais.

**Idade, custo e histórico prévio dominam o risco de readmissão — mas nenhuma variável isolada é forte.** As três features mais importantes do classificador (idade, valor da internação, número de internações anteriores) respondem por quase 78% da importância total, mas o AUC de 0,60 mostra que dados puramente administrativos têm um teto de desempenho. Prever readmissão de forma mais precisa provavelmente exige informação clínica que o SIH não traz — comorbidades detalhadas, exames, medicação em uso.

**Hospitais públicos concentram a maior pressão de ocupação.** Isso já estava embutido na forma como calibrei a capacidade de leitos simulada (meta de ocupação mais alta para gestão pública, refletindo o problema real do SUS), mas a clusterização confirma que esse recorte aparece de forma consistente nos indicadores operacionais, não só na configuração inicial.

## Métricas dos Modelos

**Previsão de ocupação (Holt-Winters, baseline)**

| Métrica | Valor |
|---|---|
| MAE | 0,057 (5,7 pontos percentuais de ocupação) |
| MAPE | 6,1% |
| Horizonte avaliado | 30 dias |

**Risco de readmissão em 30 dias**

| Modelo | AUC | Recall | Precisão | F1 |
|---|---|---|---|---|
| Regressão Logística | 0,602 | 0,579 | 0,251 | 0,350 |
| Random Forest | 0,598 | 0,527 | 0,259 | 0,347 |

**Clusterização de hospitais**

| Cluster | Perfil | Ocupação média | Readmissão | Hospitais |
|---|---|---|---|---|
| Crise aguda de leitos | Ocupação extrema + maior readmissão | 118,6% | 21,5% | 2 |
| Superlotação crônica | No limite da capacidade | 102,5% | 19,8% | 6 |
| Alta demanda sob controle | Maior volume, ocupação saudável | 82,4% | 20,5% | 4 |
| Operação equilibrada | Referência da rede | 76,3% | 20,1% | 3 |
| Baixa demanda, mortalidade mais alta | Menor ocupação, maior mortalidade | 56,3% | 20,2% | 1 |

Silhouette score do agrupamento: 0,242 — moderado, esperado para uma rede pequena (16 hospitais) com indicadores operacionais que não formam grupos totalmente estanques.

## Limitações

Os números deste relatório vêm de um dataset simulado, não de hospitais reais — servem para validar a metodologia e o pipeline, não como diagnóstico real de nenhuma rede hospitalar. O `data/README.md` documenta o caminho para substituir pelos dados reais do SIH e da WHO.

O classificador de risco de readmissão usa só variáveis administrativas. Sem comorbidades, exames e histórico clínico mais detalhado, o teto de desempenho (AUC por volta de 0,60) provavelmente não sobe muito, mesmo trocando de algoritmo.

A previsão de ocupação foi avaliada num horizonte de 30 dias sobre três anos de dados simulados — pouco para validar com confiança a componente de sazonalidade anual do Prophet, que se beneficia de vários ciclos completos de um ano.

A capacidade de leitos de cada hospital foi calibrada a partir da própria demanda simulada (Lei de Little), o que garante uma taxa de ocupação realista, mas significa que a "capacidade real" de cada unidade não é um dado independente — é uma escolha de modelagem, documentada em `src/data/gerar_dataset_simulado.py`.

## Recomendações

**Priorizar o grupo de crise aguda de leitos.** São só 2 hospitais na rede simulada, mas concentram a combinação mais perigosa: ocupação acima de 115% e a maior taxa de readmissão. Reforço de leitos ou redistribuição de pacientes para hospitais do cluster "alta demanda sob controle" (que tem volume parecido, mas ocupação saudável) é o movimento com maior retorno por esforço.

**Usar a previsão de ocupação como gatilho de alerta, não como decisão automática.** Com MAPE de 6% no baseline mais simples, a previsão já é confiável para acionar um alerta operacional de tendência de alta — mas não para decisões automáticas de remanejamento sem revisão humana, dado que picos pontuais ainda escapam do modelo.

**Tratar o risco de readmissão como uma lista de prioridade, não como um veredito.** Com AUC de 0,60, o modelo serve para ordenar quem recebe uma ligação de acompanhamento pós-alta primeiro, não para decidir sozinho quem é ou não de risco.

**Investir em dados clínicos antes de investir em um modelo mais sofisticado.** Trocar Random Forest por um algoritmo mais complexo não deve mover o AUC de forma relevante enquanto as features continuarem sendo só administrativas — o índice de comorbidades é o próximo investimento com maior retorno esperado.

## Próximos Passos

- Validar o pipeline com uma amostra real do SIH de um estado, para confirmar se as relações estatísticas do gerador simulado se sustentam
- Incluir índice de comorbidades (Charlson ou Elixhauser) na base de risco de readmissão
- Estender a avaliação da previsão de ocupação para um horizonte de 60-90 dias, onde a sazonalidade anual do Prophet deve mostrar mais vantagem sobre o baseline
- Cruzar os clusters de hospitais com dados de mobilidade de pacientes entre unidades, para simular o efeito real de uma redistribuição de leitos
- Construir um painel Power BI consumindo as tabelas processadas de `data/processed/`, com os indicadores por hospital atualizados incrementalmente

## Equipe e Período

Projeto desenvolvido como demonstração de capacidades em ciência de dados aplicada à gestão hospitalar. Dataset simulado, pipeline reproduzível, código aberto sob licença MIT.
