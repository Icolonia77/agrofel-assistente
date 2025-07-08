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

# Lógica para carregar a chave de API de forma segura, tanto localmente quanto na nuvem
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        # Tenta obter a chave dos segredos do Streamlit se o .env não funcionar (para deploy)
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("Chave de API do Google não encontrada! Configure-a no arquivo .env ou nos segredos do Streamlit.")
        st.stop()
genai.configure(api_key=api_key)


# --- FUNÇÕES DE LÓGICA (BACKEND DA APLICAÇÃO) ---

@st.cache_resource
def carregar_base_conhecimento():
    """
    Carrega o índice FAISS do disco. Se não existir, cria um novo
    a partir dos documentos na pasta 'documentos/'.
    Esta função é otimizada para o deploy no Streamlit Cloud.
    """
    CAMINHO_INDEX_FAISS = "faiss_index_agrofel" # Usamos um caminho relativo para portabilidade
    PASTA_BULAS = "documentos/"

    if os.path.exists(CAMINHO_INDEX_FAISS):
        st.info("Carregando base de conhecimento existente...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
        st.success("Base de conhecimento carregada!")
    else:
        st.warning("Base de conhecimento não encontrada. Criando uma nova (isso pode levar alguns minutos na primeira execução)...")
        
        if not os.path.exists(PASTA_BULAS) or not os.listdir(PASTA_BULAS):
            st.error(f"ERRO CRÍTICO: A pasta '{PASTA_BULAS}' não foi encontrada ou está vazia no repositório.")
            return None, None

        lista_arquivos = [f for f in os.listdir(PASTA_BULAS) if f.lower().endswith('.pdf')]
        if not lista_arquivos:
            st.error(f"Nenhum PDF encontrado na pasta '{PASTA_BULAS}'. Não é possível criar a base.")
            return None, None

        todos_documentos = []
        progress_bar = st.progress(0, text="Processando documentos...")
        for i, nome_arquivo in enumerate(lista_arquivos):
            caminho_completo = os.path.join(PASTA_BULAS, nome_arquivo)
            loader = PyPDFLoader(caminho_completo)
            paginas = loader.load()
            for pagina in paginas:
                pagina.metadata['source'] = nome_arquivo
            todos_documentos.extend(paginas)
            progress_bar.progress((i + 1) / len(lista_arquivos), text=f"Processando {nome_arquivo}")
        
        progress_bar.empty()

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

    # ETAPA 1: BUSCA VETORIAL AMPLA
    docs_relevantes = db.similarity_search(query, k=10)
    
    if not docs_relevantes:
        return "NAO_ENCONTRADO"

    # ETAPA 2: FILTRAGEM E RE-RANKEAMENTO COM LLM
    contexto_bruto = "\n\n---\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs_relevantes)])

    prompt_filtragem = f"""
    Analise a PERGUNTA do usuário e os seguintes CHUNKS de texto extraídos de bulas de produtos.
    Sua tarefa é identificar e retornar APENAS o conteúdo dos 3 chunks que são MAIS RELEVANTES e ÚTEIS para responder à pergunta.
    Não adicione nenhuma palavra sua, apenas retorne o texto exato dos chunks selecionados, separados por '---'.

    PERGUNTA: "{query}"

    CHUNKS:
    {contexto_bruto}
    """
    
    llm_filtro = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    contexto_filtrado = llm_filtro.invoke(prompt_filtragem).content

    # ETAPA 3: GERAÇÃO DA RESPOSTA FINAL
    prompt_geracao_final = f"""
    Você é um consultor especialista da Agrofel. Sua tarefa é analisar a pergunta de um agricultor
    e, com base nos TRECHOS RELEVANTES DAS BULAS, gerar uma recomendação clara e objetiva.

    PERGUNTA DO AGRICULTOR: "{query}"

    TRECHOS RELEVANTES DAS BULAS (já pré-filtrados):
    ---
    {contexto_filtrado}
    ---

    INSTRUÇÕES:
    1. Com base nos trechos, identifique e sugira até DOIS melhores produtos. Se encontrar apenas um, sugira apenas um.
    2. Para cada produto, extraia o nome exato e crie uma descrição curta e convincente, mencionando a praga/cultura.
    3. Apresente a resposta no seguinte formato, e somente neste formato:
       
       **Produto 1:** [Nome do Produto]
       **Descrição:** [Sua descrição persuasiva aqui]

       **Produto 2:** [Nome do Produto]
       **Descrição:** [Sua descrição persuasiva aqui]
    
    4. Se mesmo após a filtragem os trechos não forem suficientes para uma recomendação clara e segura, responda APENAS com: "NAO_ENCONTRADO".
    """

    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content

def notificar_vendedor(pergunta, recomendacao):
    """Simula o acionamento do vendedor."""
    print("\n--- [AÇÃO INTERNA: LEAD ENVIADO PARA VENDAS] ---")
    print(f"Cliente fez a pergunta: '{pergunta}'")
    print("Produtos sugeridos e aceitos:")
    print(recomendacao)
    print("-------------------------------------------------\n")

def encaminhar_para_humano(pergunta):
    """Simula o encaminhamento para atendimento humano."""
    print("\n--- [AÇÃO INTERNA: ATENDIMENTO TRANSFERIDO] ---")
    print(f"Cliente com a pergunta '{pergunta}' solicitou falar com um especialista.")
    print("-------------------------------------------------\n")

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
                st.success("Pedido confirmado! Seu consultor Agrofel foi notificado e dará andamento à sua solicitação.")
                st.balloons()
                st.session_state.recomendacao = ""
        
        with col2:
            if st.button("🗣️ Falar com um Humano", use_container_width=True):
                encaminhar_para_humano(st.session_state.pergunta)
                st.info("Sua solicitação foi registrada. Um especialista entrará em contato para discutir outras opções.")
                st.session_state.recomendacao = ""
