from urllib.parse import urlparse
import re

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


# Exemplo de uso
if __name__ == "__main__":
    urls = [
        "https://www.uol.com.br/futebol/noticias",
        "http://example.com/path/to/page?query=param",
        "www.test-site.org/another/path/",
        "https://subdomain.example.co.uk/path/to/resource.html",
        "invalid-url",
        "help/Python.html"
    ]
    print("=== Resultados da limpeza de URLs ===")
    for url in urls:
        cleaned_url = limpar_url(url)
        print(f"URL original: {url} -> URL limpa: {cleaned_url}")

