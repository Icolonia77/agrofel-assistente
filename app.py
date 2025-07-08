# app.py
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
st.set_page_config(page_title="Assistente Agrofel", page_icon="🌿", layout="wide")

# Carrega as variáveis de ambiente do arquivo .env (para execução local)
load_dotenv()

# Lógica para carregar a chave de API de forma segura
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("Chave de API do Google não encontrada! Configure-a no arquivo .env ou nos segredos do Streamlit.")
        st.stop()
genai.configure(api_key=api_key)


# --- FUNÇÕES DE LÓGICA (BACKEND DA APLICAÇÃO) ---

@st.cache_resource
def carregar_base_conhecimento():
    """
    Carrega ou cria o índice FAISS. Agora inclui tratamento de erro para arquivos individuais.
    """
    CAMINHO_INDEX_FAISS = "faiss_index_agrofel"
    PASTA_BULAS = "documentos/"

    if os.path.exists(CAMINHO_INDEX_FAISS):
        st.info("Carregando base de conhecimento existente...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
        st.success("Base de conhecimento carregada!")
    else:
        st.warning("Base de conhecimento não encontrada. Criando uma nova (isso pode levar alguns minutos)...")
        
        if not os.path.exists(PASTA_BULAS) or not os.listdir(PASTA_BULAS):
            st.error(f"ERRO CRÍTICO: A pasta '{PASTA_BULAS}' não foi encontrada no repositório.")
            return None, None

        lista_arquivos = [f for f in os.listdir(PASTA_BULAS) if f.lower().endswith('.pdf')]
        todos_documentos = []
        arquivos_falharam = []
        
        progress_bar = st.progress(0, text="Processando documentos...")
        for i, nome_arquivo in enumerate(lista_arquivos):
            try:
                caminho_completo = os.path.join(PASTA_BULAS, nome_arquivo)
                loader = PyPDFLoader(caminho_completo)
                paginas = loader.load()
                for pagina in paginas:
                    pagina.metadata['source'] = nome_arquivo
                todos_documentos.extend(paginas)
            except Exception as e:
                # Se um arquivo falhar, ele será adicionado à lista de falhas
                arquivos_falharam.append(f"{nome_arquivo} (Erro: {e})")
            
            progress_bar.progress((i + 1) / len(lista_arquivos), text=f"Processando {nome_arquivo}")
        
        progress_bar.empty()

        # Exibe um aviso se algum arquivo falhou na leitura
        if arquivos_falharam:
            st.error(f"Atenção: Os seguintes arquivos não puderam ser processados e foram ignorados:")
            for arquivo in arquivos_falharam:
                st.write(f"- `{arquivo}`")

        if not todos_documentos:
            st.error("Nenhum documento pôde ser processado. A base de dados não pode ser criada.")
            return None, None

        with st.spinner("Dividindo textos e criando índice vetorial..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            chunks = text_splitter.split_documents(todos_documentos)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(CAMINHO_INDEX_FAISS)
        
        st.success("Nova base de conhecimento criada com sucesso!")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
    return db, llm

def agente_especialista_recomenda(query: str, db, llm):
    """
    Usa uma lógica RAG de múltiplas etapas para encontrar e recomendar produtos.
    """
    if db is None or llm is None:
        return "Erro: A base de conhecimento não está carregada."

    docs_relevantes = db.similarity_search(query, k=10)
    if not docs_relevantes:
        return "NAO_ENCONTRADO"

    contexto_bruto = "\n\n---\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs_relevantes)])
    prompt_filtragem = f"""
    Analise a PERGUNTA do usuário e os seguintes CHUNKS de texto.
    Sua tarefa é retornar APENAS o conteúdo dos 3 chunks que são MAIS RELEVANTES para responder à pergunta, separados por '---'.
    PERGUNTA: "{query}"
    CHUNKS:
    {contexto_bruto}
    """
    llm_filtro = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    contexto_filtrado = llm_filtro.invoke(prompt_filtragem).content

    prompt_geracao_final = f"""
    Você é um consultor especialista da Agrofel. Com base nos TRECHOS RELEVANTES DAS BULAS, gere uma recomendação.
    PERGUNTA DO AGRICULTOR: "{query}"
    TRECHOS RELEVANTES DAS BULAS:
    ---
    {contexto_filtrado}
    ---
    INSTRUÇÕES:
    1. Sugira até DOIS produtos. Se encontrar apenas um, sugira apenas um.
    2. Extraia o nome exato do produto e crie uma descrição curta.
    3. Formato:
       **Produto 1:** [Nome do Produto]
       **Descrição:** [Sua descrição]
    4. Se os trechos não forem suficientes, responda APENAS com: "NAO_ENCONTRADO".
    """
    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content

def notificar_vendedor(pergunta, recomendacao):
    print(f"\n--- LEAD PARA VENDAS: {pergunta} ---")
    print(recomendacao)

def encaminhar_para_humano(pergunta):
    print(f"\n--- TRANSFERÊNCIA PARA ATENDIMENTO: {pergunta} ---")

# --- INTERFACE DO USUÁRIO (STREAMLIT) ---
st.title("🌿 Assistente de Campo Agrofel")
st.markdown("Bem-vindo! Descreva seu problema com pragas na lavoura e encontrarei a melhor solução para você.")

db, llm = carregar_base_conhecimento()

if 'recomendacao' not in st.session_state:
    st.session_state.recomendacao = ""
if 'pergunta' not in st.session_state:
    st.session_state.pergunta = ""

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
        st.error("A base de conhecimento não pôde ser carregada. Verifique os logs do terminal.")

if st.session_state.recomendacao:
    if "NAO_ENCONTRADO" in st.session_state.recomendacao:
        st.warning("Não encontrei produtos específicos para sua solicitação em nossa base de dados.")
        st.info("Gostaria de falar com um de nossos consultores para obter ajuda personalizada?")
        if st.button("Sim, solicitar contato de um especialista"):
            encaminhar_para_humano(st.session_state.pergunta)
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
                notificar_vendedor(st.session_state.pergunta, st.session_state.recomendacao)
                st.success("Pedido confirmado! Seu consultor Agrofel foi notificado.")
                st.balloons()
                st.session_state.recomendacao = ""
        with col2:
            if st.button("🗣️ Falar com um Humano", use_container_width=True):
                encaminhar_para_humano(st.session_state.pergunta)
                st.info("Sua solicitação foi registrada. Um especialista entrará em contato.")
                st.session_state.recomendacao = ""
