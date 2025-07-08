# app.py (Versão Final com Transformação de Consulta)
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Imports para LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        st.error(f"ERRO CRÍTICO: A base de conhecimento pré-construída ('{CAMINHO_INDEX_FAISS}') não foi encontrada.")
        return None, None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
        return db, llm
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar a base de conhecimento: {e}")
        return None, None

def agente_especialista_recomenda(query: str, db, llm):
    """
    Implementa a lógica RAG com Transformação de Consulta para maior robustez.
    """
    if db is None or llm is None:
        return "Erro: A base de conhecimento não está carregada."

    # ETAPA 1: TRANSFORMAÇÃO DA CONSULTA
    # Usamos um LLM para gerar múltiplas consultas de busca a partir da pergunta original.
    template_transformacao = """Você é um especialista em agronomia. Sua tarefa é transformar a pergunta de um utilizador numa lista de 3 consultas de busca otimizadas para uma base de dados vetorial.
    As consultas devem ser concisas e variadas para cobrir diferentes aspetos da pergunta.
    Responda apenas com as consultas, uma por linha.

    Pergunta Original: {pergunta}

    Consultas de Busca:
    """
    prompt_transformacao = ChatPromptTemplate.from_template(template_transformacao)
    
    cadeia_transformacao = prompt_transformacao | llm | StrOutputParser()
    consultas_geradas = cadeia_transformacao.invoke({"pergunta": query}).strip().split('\n')

    st.sidebar.subheader("Consultas de Busca Geradas")
    st.sidebar.write(consultas_geradas)

    # ETAPA 2: BUSCA AUMENTADA
    # Fazemos uma busca para cada consulta gerada e juntamos os resultados.
    todos_chunks = []
    for consulta in consultas_geradas:
        todos_chunks.extend(db.similarity_search(consulta, k=3))
    
    # Removemos duplicados, se houver
    unique_chunks = {doc.page_content: doc for doc in todos_chunks}.values()

    if not unique_chunks:
        return "NAO_ENCONTRADO"

    contexto_final = "\n\n---\n\n".join([doc.page_content for doc in unique_chunks])

    # ETAPA 3: GERAÇÃO DA RESPOSTA FINAL
    # A lógica de geração permanece a mesma, mas agora com um contexto muito melhor.
    prompt_geracao_final = f"""Você é um consultor especialista da Agrofel. Com base nos TRECHOS RELEVANTES DAS BULAS, gere uma recomendação clara e objetiva.

    PERGUNTA ORIGINAL DO AGRICULTOR: "{query}"

    TRECHOS RELEVANTES DAS BULAS (obtidos de múltiplas buscas):
    ---
    {contexto_final}
    ---
    INSTRUÇÕES:
    1. Sugira até DOIS produtos que melhor respondam à pergunta. Se encontrar apenas um, sugira apenas um.
    2. Extraia o nome exato do produto e crie uma descrição curta e convincente.
    3. Formato:
       **Produto 1:** [Nome do Produto]
       **Descrição:** [Sua descrição]

       **Produto 2:** [Nome do Produto]
       **Descrição:** [Sua descrição]
    4. Se os trechos não forem suficientes, responda APENAS com: "NAO_ENCONTRADO".
    """
    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content

# --- INTERFACE DO USUÁRIO ---
st.title("🌿 Assistente de Campo Agrofel")
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
    pergunta_usuario = st.text_area("Qual praga está afetando sua lavoura e em qual cultura?", height=100, placeholder="Ex: Capim amargoso na soja")
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
    else:
        st.subheader("Encontrei estas sugestões para você:")
        st.markdown(st.session_state.recomendacao)
    
    # Lógica dos botões de ação (simplificada para clareza)
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

