


## Limpeza Estruturada de URLs
partes de uma url:
- protocolo (http) 
- subdominio (www)
- dominio principal (yahoo.com)
- o caminho (/new/footbal)
- parametros de busca

para classificar uma url corretamente, precisamos extrair apenas o texto relevantes e remover ruidos (como http, https e www).

vamos usar a biblioteca nativa do python urllib.parse

## Tokenização Customizada e Vetorização TF-IDF
### Feature Engineering (Vetorização)

Os algoritmos de ML não entendem palavras, apenas números. Precisamos transformar nossas strings limpas em uma matriz numerica.
TF-IDF (Term Frequency-Unverse Document Frequency) é a melhor tecnica, associada a uma tokenização customizada para URLs

Em textos normais, as palavras são separadas por espaços. Em URL termos importantes muitas vezes ficam colados por barras ou traços

O TF-IDF analisa duas coisas
**Term Frequency (TF)** Quão frequente uma palavra é dentro de uma unica url (se 'sports' aparece muito entao ela é importante ali).

**Inverse Document Frequency (IDF)** Quão comum a palavra é em todas urls do seu banco de dados. Se a palavra "com" ou "www" aparece em absolutamente todas as urls, o IDF joga o peso dela para baixo, posi ela nao ajuda a diferenciar uma url de entretenimento de uma de finanças.

## Treinamento e Persistencia do Modelo (salvando em Disco)

Em pipelines de engenharia de dados (especialmente fluxos de MLOps), voce **nunca** deve treinar o modelo na hora de fazer a classificacao de producao. o correto eh treinar o modelo de uma unica vez, salva-lo em um arquivo fisico (artefato) e depois carregalo de forma ultra rapida no pipeline para classificar novos dados.


Usamos o algoritmo Naive Bayes (MultinominalNB) que eh extremamento leve, rapido e perfeito para classificacao de texto, junto com a biblioteca joblib para salvar os artefatos.

## Execucao

Execute `uv run url-classification` para treinar o modelo, salvar os artefatos no diretorio atual e classificar as URLs de exemplo. A importacao do pacote disponibiliza `limpar_url`, `treinar_modelo` e `classificar_urls` sem iniciar o treinamento automaticamente.


## Construindo o pipeline de Dados (Camada de Oroducao)

Em cenarios reais de engenharia de dados, voce nao roda o treino e a classificacao no mesmo script. O modelo eh consumido por um precesso em lote (batch processing) que:
    -1 le os dados brutos de uma origem 
    -2 carrega os artefatos salvos (.pkl) sem reexecutar treinos
    -3 processa o resultado final estruturado em um destino para consumo analitico