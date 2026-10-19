import sqlite3
from typing import Any

import joblib
import pandas as pd
from main import limpar_url


# configuracoes de Producao

MODEL_PATH = "modelo_classificador_url.pkl"
VECTORIZER_PATH = "vetorizador_tfidf.pkl"
DB_NAME = "analytics.db"


def carregar_artefatos_ml() -> tuple[Any, Any]:
    """Carrega o modelo e o vetorizador do disco."""
    try:
        modelo = joblib.load(MODEL_PATH)
        vetorizador = joblib.load(VECTORIZER_PATH)
        return modelo, vetorizador
    except FileNotFoundError as e:
        raise RuntimeError(f"Arquivo de artefatos não encontrado: {e}") from e


def simular_ingestao_dados() -> pd.DataFrame:
    """Simula a ingestão de dados de URLs para classificação."""
    dados_brutos = {
        "log_id": [1, 2, 3, 4, 5],
        "user_id": ["user_01", "user_02", "user_01", "user_03", "user_04"],
        "timestamp": [
            "2026-10-19 10:00:00",
            "2026-10-19 10:01:23",
            "2026-10-19 10:02:11",
            "2026-10-19 10:03:00",
            "2026-10-19 10:05:45",
        ],
        "url_bruta": [
            "https://investing.com",
            "http://uol.com.br",
            "https://bbc.com",
            "https://github.com",
            "www.example.org/downloads/document.pdf",
        ],
    }
    return pd.DataFrame(dados_brutos)


def executar_transformacao_e_predicao(
    df: pd.DataFrame, modelo: Any, vetorizador: Any
) -> pd.DataFrame:
    """Aplica a limpeza e classificação das URLs no DataFrame."""
    # Cria uma copia do DataFrame para evitar modificar o original
    df_processado = df.copy()

    # Limpa as URLs
    urls_limpas = df_processado["url_bruta"].apply(limpar_url).tolist()

    # Transforma o lote de textos limpos na matriz numérica esparsa
    x_features = vetorizador.transform(urls_limpas)

    # Classifica o bloco inteiro de uma só vez.
    df_processado["categoria_predita"] = modelo.predict(x_features)

    return df_processado

def carregar_dados_no_banco(df: pd.DataFrame) -> None:
    """Carrega o DataFrame processado em um banco de dados SQLite."""
    with sqlite3.connect(DB_NAME) as conn:
        df.to_sql("url_classification_results", conn, if_exists="append", index=False)


def main() -> None:
    print("🚀 [INÍCIO DO PIPELINE] Inicializando componentes...")
    modelo, vetorizador = carregar_artefatos_ml()

    print("📥 [INGESTÃO] Coletando lote de URLs pendentes...")
    df_logs = simular_ingestao_dados()

    print("🧠 [PROCESSAMENTO] Higienizando strings e aplicando inferência de ML...")
    df_final = executar_transformacao_e_predicao(df_logs, modelo, vetorizador)

    print("📤 [CARGA] Persistindo os dados classificados no banco analítico...")
    carregar_dados_no_banco(df_final)

    print("✅ [FIM DO PIPELINE] Processo concluído com sucesso!")

    with sqlite3.connect(DB_NAME) as conn:
        resultado_banco = pd.read_sql(
            "SELECT log_id, url_bruta, categoria_predita "
            "FROM url_classification_results",
            conn,
        )
        print("\n=== DADOS RECUPERADOS DA TABELA DO BANCO DE DADOS ===")
        print(resultado_banco.to_string())


if __name__ == "__main__":
    main()