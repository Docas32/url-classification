import re
from urllib.parse import urlparse

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

TRAINING_DATA = (
    ("https://bloomberg.com", "Finance"),
    ("https://www.nba.com", "Sports"),
    ("https://www.uol.com.br/futebol/noticias", "Sports"),
    ("https://www.yahoo.com", "News"),
    ("https://www.cnn.com", "News"),
    ("http://www.example.com/path/to/page?query=param", "Other"),
    ("https://subdomain.example.org/another/path/", "Other"),
    ("www.testsite.com/some-page.html", "Other"),
    ("http://www.sample-site.net/index.php?id=123&cat=456", "Other"),
    ("https://www.espn.com/soccer", "Sports"),
    ("https://www.bbc.com/news/world", "News"),
    ("https://www.investing.com/markets", "Finance"),
)

NEW_URLS = (
    "https://www.espn.com",
    "https://www.bbc.com/news",
    "https://www.investing.com",
    "https://www.example.org/some/path",
)

MODEL_PATH = "modelo_classificador_url.pkl"
VECTORIZER_PATH = "vetorizador_tfidf.pkl"


def limpar_url(url: str) -> str:
    """Remove o protocolo, www e pontuação, preservando palavras e números."""
    url = url.lower().strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed_url = urlparse(url)
    netloc = parsed_url.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]

    url_combinada = netloc + parsed_url.path
    url_limpa = re.sub(r"[^a-zA-Z0-9\s]", " ", url_combinada)
    return " ".join(url_limpa.split())


def treinar_modelo() -> tuple[MultinomialNB, TfidfVectorizer]:
    urls, categorias = zip(*TRAINING_DATA, strict=True)
    vetorizador = TfidfVectorizer(ngram_range=(1, 2))
    x_train = vetorizador.fit_transform([limpar_url(url) for url in urls])
    modelo = MultinomialNB().fit(x_train, categorias)
    return modelo, vetorizador


def classificar_urls(
    urls: tuple[str, ...] | list[str],
    modelo: MultinomialNB,
    vetorizador: TfidfVectorizer,
) -> list[str]:
    if not urls:
        return []

    x_urls = vetorizador.transform([limpar_url(url) for url in urls])
    return modelo.predict(x_urls).tolist()


def main() -> None:
    modelo, vetorizador = treinar_modelo()
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(vetorizador, VECTORIZER_PATH)

    modelo_carregado = joblib.load(MODEL_PATH)
    vetorizador_carregado = joblib.load(VECTORIZER_PATH)
    predicoes = classificar_urls(NEW_URLS, modelo_carregado, vetorizador_carregado)

    print("=== MODELO TREINADO COM SUCESSO ===")
    print(f"=== MODELO E VETORIZADOR SALVOS: {MODEL_PATH}, {VECTORIZER_PATH} ===")
    for url, categoria in zip(NEW_URLS, predicoes, strict=True):
        print(f"URL: {url} => Categoria Predita: {categoria}")
