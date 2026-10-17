


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

