from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def limpar_url(url):
    """
    Remove protocolos, www, pontuação e extrai os componentes de texto da URL"""


    # 1. Converte para minusculo para manter consistência
    url = url.lower().strip()

    # 2. Garante que a URL tenha um protocolo para o ulrparse funcionar corretamente
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        parsed_url = urlparse(url)
        #Extrai o domínio (ex: ://uol.com.br) e o caminho (ex: /futebol/noticias)
        netloc = parsed_url.netloc
        path = parsed_url.path

        #  Remove 'www.' do domínio
        if netloc.startswith('www.'):
            netloc = netloc[4:]

        # Junta o domínio e o caminho limpos
        url_combinada = netloc + path

        # 3. Limpeza com Expressões Regulares (regex)
        # Substitui caracteres especiais, barras, tracos e pontos por espaço em branco
        url_limpa = re.sub(re.compile(r'[^a-zA-Z\s]'), ' ', url_combinada)

        # Remove espaços extras duplicados
        url_limpa = ' '.join(url_limpa.split())

        return url_limpa
    except Exception as e:
        print(f"Erro ao processar a URL: {e}")
        return ""


# lista de URLs brutas para teste
urls_brutas = [
    "https://www.uol.com.br/futebol/noticias",
    "http://www.example.com/path/to/page?query=param",
    "https://subdomain.example.org/another/path/",
    "www.testsite.com/some-page.html",
    "http://www.sample-site.net/index.php?id=123&cat=456",
    "https://www.another-example.com/path/to/resource?search=query&sort=asc",
    "http://www.example.com/path/to/page?query=param#section",
    "https://www.example.com/path/to/page?query=param&another=value",
    "http://www.example.com/path/to/page?query=param&another=value#section",
    "https://www.example.com/path/to/page?query=param&another=value#section2"
]

# 1. Aplicando o pipeline de limpeza de URLs em lote

urls_limpas = [limpar_url(url) for url in urls_brutas]

# 2. Inicializando o Vectorizer TF-IDF
# Usamos ngram_range=(1, 2) para capturar palavras isoladas (ex "nba") e pares de palavras (ex "yahoo com")
vetorizador = TfidfVectorizer(ngram_range=(1, 2), max_features= 15)

# 3. Ajustando e transformando as strings em matriz numerica (vetorizacão)
matriz_tfidf = vetorizador.fit_transform(urls_limpas)

# --- Verificando os resultados ---
# Convertendo a matriz TF-IDF para um DataFrame do Pandas para melhor visualização
nome_dos_recuros = vetorizador.get_feature_names_out()
df_features = pd.DataFrame(matriz_tfidf.toarray(), columns=nome_dos_recuros, index=urls_brutas)

# Exemplo de uso
if __name__ == "__main__":
    
    print("=== MATRIZ TF-IDF GERADA (RECURSOS NUMÉRICOS)    ===")
    print(df_features.round(3))  # Arredondando para 3 casas decimais para melhor visualização


