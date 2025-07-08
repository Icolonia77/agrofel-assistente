# 1_Criar_Base_Vetorial.py
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Carrega as variáveis de ambiente (sua chave de API) do arquivo .env
load_dotenv()

# Configura a API do Google
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("A chave GOOGLE_API_KEY não foi encontrada no arquivo .env")
genai.configure(api_key=api_key)

# --- Definição dos Caminhos Locais para Windows ---
# Caminho base do seu projeto. Use barras normais.
CAMINHO_BASE_PROJETO = "C:/Users/Usuario/Documents/Agrofel/"
# Caminho completo para a pasta de documentos
PASTA_BULAS = os.path.join(CAMINHO_BASE_PROJETO, "documentos/")
# Caminho completo para salvar o índice vetorial
CAMINHO_INDEX_FAISS = os.path.join(CAMINHO_BASE_PROJETO, "faiss_index_agrofel")

def criar_base_de_conhecimento():
    """
    Processa todos os PDFs na pasta 'documentos' e cria/salva o índice FAISS.
    """
    if not os.path.exists(PASTA_BULAS) or not os.listdir(PASTA_BULAS):
        print(f"ERRO: A pasta '{PASTA_BULAS}' não existe ou está vazia.")
        return

    print(f"Iniciando processamento dos PDFs em '{PASTA_BULAS}'...")
    inicio = time.time()

    lista_arquivos = [f for f in os.listdir(PASTA_BULAS) if f.lower().endswith('.pdf')]
    if not lista_arquivos:
        print(f"Nenhum PDF encontrado em '{PASTA_BULAS}'.")
        return

    todos_documentos = []
    for nome_arquivo in lista_arquivos:
        caminho_completo = os.path.join(PASTA_BULAS, nome_arquivo)
        print(f"Processando: {nome_arquivo}")
        loader = PyPDFLoader(caminho_completo)
        paginas = loader.load()
        for pagina in paginas:
            pagina.metadata['source'] = nome_arquivo
        todos_documentos.extend(paginas)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(todos_documentos)

    print(f"\n{len(lista_arquivos)} PDFs processados, {len(chunks)} chunks de texto criados.")
    print("Inicializando modelo de embeddings e criando índice FAISS...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db_vetorial = FAISS.from_documents(chunks, embeddings)
    db_vetorial.save_local(CAMINHO_INDEX_FAISS)

    fim = time.time()
    print(f"\nSUCESSO! Base de conhecimento criada em '{CAMINHO_INDEX_FAISS}'.")
    print(f"Tempo de execução: {round(fim - inicio, 2)}s.")

if __name__ == "__main__":
    criar_base_de_conhecimento()