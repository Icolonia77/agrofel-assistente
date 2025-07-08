# app.py (Versão Final de Produção com Debug)
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Imports para LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# --- CONFIGURAÇÃO INICIAL DA PÁGINA E VARIÁVEIS DE AMBIENTE ---
st.set_page_config(page_title="Assistente Agrofel", page_icon="🌿", layout="wide")

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

@st.cache_resource(show_spinner="A carregar base de conhecimento...")
def carregar_base_conhecimento():
    """
    Carrega o índice FAISS pré-construído do repositório.
    """
    CAMINHO_INDEX_FAISS = "faiss_index_agrofel"

    if not os.path.exists(CAMINHO_INDEX_FAISS):
        st.error(f"ERRO CRÍTICO: A base de conhecimento pré-construída ('{CAMINHO_INDEX_FAISS}') não foi encontrada no repositório.")
        return None, None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
        return db, llm
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar a base de conhecimento: {e}")
        return None, None

def agente_especialista_recomenda(query: str, db, llm):
    """
    Usa a lógica RAG de múltiplas etapas para encontrar e recomendar produtos.
    """
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

# --- O RESTANTE DO CÓDIGO DA INTERFACE PERMANECE O MESMO ---

st.title("🌿 Assistente de Campo Agrofel")

# --- PAINEL DE DEBUG NA BARRA LATERAL ---
with st.sidebar:
    st.header("Opções de Desenvolvedor")
    if st.button("🚨 Limpar Cache da Aplicação"):
        st.cache_resource.clear()
        st.success("Cache limpo! A aplicação irá recarregar a base de dados.")
        st.rerun()

st.markdown("Bem-vindo! Descreva seu problema com pragas na lavoura e encontrarei a melhor solução para você.")

db, llm = carregar_base_conhecimento()

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
    if "NAO_ENCONTRADO" in st.session_state.recomendacao:
        st.warning("Não encontrei produtos específicos para sua solicitação em nossa base de dados.")
        st.info("Gostaria de falar com um de nossos consultores para obter ajuda personalizada?")
        if st.button("Sim, solicitar contato de um especialista"):
            st.success("Sua solicitação foi enviada! Um consultor Agrofel entrará em contato em breve.")
            st.session_state.recomendacao = ""
    else:
        st.subheader("Encontrei estas sugestões para você:")
        st.markdown(st.session_state.recomendacao)
        st.markdown("---")
        st.markdown("O que você gostaria de fazer?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirmar Pedido", type="primary", use_container_width=True):
                st.success("Pedido confirmado! Seu consultor Agrofel foi notificado.")
                st.balloons()
                st.session_state.recomendacao = ""
        with col2:
            if st.button("🗣️ Falar com um Humano", use_container_width=True):
                st.info("Sua solicitação foi registrada. Um especialista entrará em contato.")
                st.session_state.recomendacao = ""
