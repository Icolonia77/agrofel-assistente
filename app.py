# app.py (Versão de Diagnóstico)
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Imports para LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURAÇÃO INICIAL DA PÁGINA E VARIÁVEIS DE AMBIENTE ---
st.set_page_config(page_title="Assistente Agrofel (Debug)", page_icon="🐞", layout="wide")

# Lógica para carregar a chave de API de forma segura
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("Chave de API do Google não encontrada! Configure-a no arquivo .env ou nos segredos do Streamlit.")
        st.stop()
genai.configure(api_key=api_key)


# --- FUNÇÕES DE LÓGICA (BACKEND DA APLICAÇÃO) ---

@st.cache_resource(show_spinner="A carregar/criar a base de conhecimento...")
def carregar_base_conhecimento():
    """
    Carrega ou cria o índice FAISS. Retorna o DB, o LLM e metadados de diagnóstico.
    """
    CAMINHO_INDEX_FAISS = "faiss_index_agrofel"
    PASTA_BULAS = "documentos/"
    
    diagnostico = {
        "arquivos_encontrados": [],
        "arquivos_falharam": [],
        "total_chunks": 0
    }

    if not os.path.exists(PASTA_BULAS) or not os.listdir(PASTA_BULAS):
        st.error(f"ERRO CRÍTICO: A pasta '{PASTA_BULAS}' não foi encontrada ou está vazia no repositório.")
        return None, None, diagnostico
    
    diagnostico["arquivos_encontrados"] = sorted([f for f in os.listdir(PASTA_BULAS) if f.lower().endswith('.pdf')])

    if os.path.exists(CAMINHO_INDEX_FAISS):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
    else:
        todos_documentos = []
        for nome_arquivo in diagnostico["arquivos_encontrados"]:
            try:
                caminho_completo = os.path.join(PASTA_BULAS, nome_arquivo)
                loader = PyPDFLoader(caminho_completo)
                paginas = loader.load()
                for pagina in paginas:
                    pagina.metadata['source'] = nome_arquivo
                todos_documentos.extend(paginas)
            except Exception as e:
                diagnostico["arquivos_falharam"].append(f"{nome_arquivo} (Erro: {e})")

        if not todos_documentos:
            st.error("Nenhum documento pôde ser processado. A base de dados não pode ser criada.")
            return None, None, diagnostico

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_documents(todos_documentos)
        diagnostico["total_chunks"] = len(chunks)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(CAMINHO_INDEX_FAISS)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
    return db, llm, diagnostico

def agente_especialista_recomenda(query: str, db, llm):
    # Esta função permanece a mesma da versão anterior
    if db is None or llm is None: return "Erro: A base de conhecimento não está carregada."
    docs_relevantes = db.similarity_search(query, k=10)
    if not docs_relevantes: return "NAO_ENCONTRADO"
    contexto_bruto = "\n\n---\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs_relevantes)])
    prompt_filtragem = f"""Analise a PERGUNTA do usuário e os seguintes CHUNKS de texto. Sua tarefa é retornar APENAS o conteúdo dos 3 chunks que são MAIS RELEVANTES para responder à pergunta, separados por '---'.\n\nPERGUNTA: "{query}"\n\nCHUNKS:\n{contexto_bruto}"""
    llm_filtro = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    contexto_filtrado = llm_filtro.invoke(prompt_filtragem).content
    prompt_geracao_final = f"""Você é um consultor especialista da Agrofel. Com base nos TRECHOS RELEVANTES DAS BULAS, gere uma recomendação.\n\nPERGUNTA DO AGRICULTOR: "{query}"\n\nTRECHOS RELEVANTES DAS BULAS:\n---\n{contexto_filtrado}\n---\nINSTRUÇÕES:\n1. Sugira até DOIS produtos. Se encontrar apenas um, sugira apenas um.\n2. Extraia o nome exato do produto e crie uma descrição curta.\n3. Formato:\n   **Produto 1:** [Nome do Produto]\n   **Descrição:** [Sua descrição]\n4. Se os trechos não forem suficientes, responda APENAS com: "NAO_ENCONTRADO"."""
    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content

# --- INTERFACE DO USUÁRIO (STREAMLIT) ---

st.title("🌿 Assistente de Campo Agrofel")
st.markdown("Bem-vindo! Descreva seu problema com pragas na lavoura e encontrarei a melhor solução para você.")

db, llm, info_diagnostico = carregar_base_conhecimento()

# --- PAINEL DE DIAGNÓSTICO ---
with st.expander("🐞 Informações de Diagnóstico (Ambiente de Produção)"):
    st.write("**Ficheiros PDF Encontrados no Repositório:**", len(info_diagnostico["arquivos_encontrados"]))
    st.json(info_diagnostico["arquivos_encontrados"])
    
    if info_diagnostico["arquivos_falharam"]:
        st.error("**Ficheiros que Falharam ao Processar:**", len(info_diagnostico["arquivos_falharam"]))
        st.json(info_diagnostico["arquivos_falharam"])
    else:
        st.success("Todos os ficheiros foram processados com sucesso.")

    if info_diagnostico["total_chunks"] > 0:
        st.write("**Total de Chunks de Texto Criados:**", info_diagnostico["total_chunks"])

    st.info("Use o botão abaixo para testar a busca pela consulta específica que está a falhar.")
    if st.button("Executar Teste de Diagnóstico para 'Capim Amargoso na Soja'"):
        with st.spinner("A executar teste..."):
            query_teste = "Qual produto usar para Capim-amargoso na cultura da soja?"
            docs_teste = db.similarity_search_with_score(query_teste, k=5)
            st.subheader("Resultado da Busca por Similaridade (Etapa 1)")
            if not docs_teste:
                st.error("A busca não retornou NENHUM resultado.")
            else:
                for i, (doc, score) in enumerate(docs_teste):
                    st.write(f"**Chunk {i+1} (Score: {score:.4f})**")
                    st.write(f"Fonte: `{doc.metadata.get('source', 'N/D')}`")
                    st.code(doc.page_content, language="text")

# --- Lógica Principal da Aplicação ---
# (O restante do código permanece o mesmo)
if 'recomendacao' not in st.session_state: st.session_state.recomendacao = ""
if 'pergunta' not in st.session_state: st.session_state.pergunta = ""

with st.form("pergunta_form"):
    pergunta_usuario = st.text_area("Qual praga está afetando sua lavoura e em qual cultura?", height=100, placeholder="Ex: Lagarta do cartucho no milho")
    submitted = st.form_submit_button("Buscar Sugestões")

if submitted and pergunta_usuario:
    if db is not None:
        with st.spinner("Analisando as melhores soluções para você..."):
            st.session_state.pergunta = pergunta_usuario
            recomendacao_gerada = agente_especialista_recomenda(pergunta_usuario, db, llm)
            st.session_state.recomendacao = recomendacao_gerada
    else:
        st.error("A base de conhecimento não pôde ser carregada.")

if st.session_state.recomendacao:
    # (A lógica de exibição da resposta permanece a mesma)
    if "NAO_ENCONTRADO" in st.session_state.recomendacao:
        st.warning("Não encontrei produtos específicos para sua solicitação em nossa base de dados.")
    else:
        st.subheader("Encontrei estas sugestões para você:")
        st.markdown(st.session_state.recomendacao)
