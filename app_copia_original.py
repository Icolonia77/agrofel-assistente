# app.py
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# --- CONFIGURAÇÃO INICIAL DA PÁGINA E VARIÁVEIS DE AMBIENTE ---
st.set_page_config(page_title="Assistente Agrofel", page_icon="🌿", layout="wide")

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Exibe um erro na interface do Streamlit se a chave não for encontrada
    st.error("Chave de API do Google não encontrada! Verifique seu arquivo .env")
    st.stop()
genai.configure(api_key=api_key)

# Caminho completo para onde o índice FAISS está salvo no seu ambiente local
CAMINHO_INDEX_FAISS = "C:/Users/Usuario/Documents/Agrofel/faiss_index_agrofel"

# --- FUNÇÕES DE LÓGICA (BACKEND DA APLICAÇÃO) ---

@st.cache_resource
def carregar_base_conhecimento():
    """
    Carrega o índice FAISS do disco e inicializa o modelo de linguagem.
    Usa o cache do Streamlit para não recarregar a cada interação do usuário.
    """
    if not os.path.exists(CAMINHO_INDEX_FAISS):
        st.error(f"A base de conhecimento (índice FAISS) não foi encontrada em '{CAMINHO_INDEX_FAISS}'. Execute o script '1_Criar_Base_Vetorial.py' primeiro.")
        return None, None
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
        # Usamos um modelo mais capaz para a geração final
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
        return db, llm
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar a base de conhecimento: {e}")
        return None, None

def agente_especialista_recomenda(query: str, db, llm):
    """
    Usa uma lógica RAG de múltiplas etapas para encontrar e recomendar produtos.
    1. Busca inicial ampla para capturar qualquer informação relevante.
    2. Filtragem e Re-rankeamento com um LLM para limpar os dados.
    3. Geração da resposta final com um LLM mais robusto.
    """
    if db is None or llm is None:
        return "Erro: A base de conhecimento não está carregada."

    # ETAPA 1: BUSCA VETORIAL AMPLA
    # Buscamos mais documentos (k=10) para aumentar a chance de encontrar algo relevante, mesmo que "sujo".
    docs_relevantes = db.similarity_search(query, k=10)
    
    if not docs_relevantes:
        return "NAO_ENCONTRADO"

    # ETAPA 2: FILTRAGEM E RE-RANKEAMENTO COM LLM
    # Criamos um contexto inicial com todos os chunks encontrados para o LLM limpar.
    contexto_bruto = "\n\n---\n\n".join([f"Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs_relevantes)])

    # Prompt para o LLM atuar como um filtro inteligente.
    prompt_filtragem = f"""
    Analise a PERGUNTA do usuário e os seguintes CHUNKS de texto extraídos de bulas de produtos.
    Sua tarefa é identificar e retornar APENAS o conteúdo dos 3 chunks que são MAIS RELEVANTES e ÚTEIS para responder à pergunta.
    Não adicione nenhuma palavra sua, apenas retorne o texto exato dos chunks selecionados, separados por '---'.

    PERGUNTA: "{query}"

    CHUNKS:
    {contexto_bruto}
    """
    
    # Usamos um modelo rápido e eficiente para a tarefa de filtragem.
    llm_filtro = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    contexto_filtrado = llm_filtro.invoke(prompt_filtragem).content

    # ETAPA 3: GERAÇÃO DA RESPOSTA FINAL
    # Agora usamos o contexto limpo e filtrado para gerar a resposta final ao usuário.
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

    # Usamos o modelo principal para a geração da resposta final de alta qualidade.
    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content

def notificar_vendedor(pergunta, recomendacao):
    """Simula o acionamento do vendedor."""
    print("\n--- [AÇÃO INTERNA: LEAD ENVIADO PARA VENDAS] ---")
    print(f"Cliente fez a pergunta: '{pergunta}'")
    print("Produtos sugeridos e aceitos:")
    print(recomendacao)
    print("-------------------------------------------------\n")
    # Em produção: aqui iria o código para enviar email, chamar uma API do CRM, etc.

def encaminhar_para_humano(pergunta):
    """Simula o encaminhamento para atendimento humano."""
    print("\n--- [AÇÃO INTERNA: ATENDIMENTO TRANSFERIDO] ---")
    print(f"Cliente com a pergunta '{pergunta}' solicitou falar com um especialista.")
    print("-------------------------------------------------\n")
    # Em produção: aqui iria o código para criar um ticket em um sistema de suporte.

# --- INTERFACE DO USUÁRIO (STREAMLIT) ---

st.title("🌿 Assistente de Campo Agrofel")
st.markdown("Bem-vindo! Descreva seu problema com pragas na lavoura e encontrarei a melhor solução para você.")

# Carrega a base de conhecimento uma vez e a mantém em cache.
db, llm = carregar_base_conhecimento()

# Inicializa o estado da sessão para controlar o fluxo da conversa.
if 'recomendacao' not in st.session_state:
    st.session_state.recomendacao = ""
if 'pergunta' not in st.session_state:
    st.session_state.pergunta = ""

# Formulário de pergunta para evitar que a página recarregue a cada letra digitada.
with st.form("pergunta_form"):
    pergunta_usuario = st.text_area("Qual praga está afetando sua lavoura e em qual cultura?", height=100, placeholder="Ex: Lagarta do cartucho no milho")
    submitted = st.form_submit_button("Buscar Sugestões")

if submitted and pergunta_usuario:
    # Apenas executa a busca se a base de dados foi carregada corretamente.
    if db is not None:
        with st.spinner("Analisando as melhores soluções para você..."):
            st.session_state.pergunta = pergunta_usuario
            recomendacao_gerada = agente_especialista_recomenda(pergunta_usuario, db, llm)
            st.session_state.recomendacao = recomendacao_gerada
    else:
        # Mensagem de erro se a base não pôde ser carregada.
        st.error("A base de conhecimento não pôde ser carregada. Verifique os logs do terminal.")

# Lógica de Exibição dos Resultados e Ações
if st.session_state.recomendacao:
    if "NAO_ENCONTRADO" in st.session_state.recomendacao:
        st.warning("Não encontrei produtos específicos para sua solicitação em nossa base de dados.")
        st.info("Gostaria de falar com um de nossos consultores para obter ajuda personalizada?")
        if st.button("Sim, solicitar contato de um especialista"):
            encaminhar_para_humano(st.session_state.pergunta)
            st.success("Sua solicitação foi enviada! Um consultor Agrofel entrará em contato em breve.")
            st.session_state.recomendacao = "" # Limpa o estado para uma nova consulta
    else:
        st.subheader("Encontrei estas sugestões para você:")
        # Usamos st.markdown para renderizar a formatação em negrito.
        st.markdown(st.session_state.recomendacao)
        
        st.markdown("---")
        st.markdown("O que você gostaria de fazer?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Confirmar Pedido", type="primary", use_container_width=True):
                notificar_vendedor(st.session_state.pergunta, st.session_state.recomendacao)
                st.success("Pedido confirmado! Seu consultor Agrofel foi notificado e dará andamento à sua solicitação.")
                st.balloons()
                st.session_state.recomendacao = "" # Limpa o estado
        
        with col2:
            if st.button("🗣️ Falar com um Humano", use_container_width=True):
                encaminhar_para_humano(st.session_state.pergunta)
                st.info("Sua solicitação foi registrada. Um especialista entrará em contato para discutir outras opções.")
                st.session_state.recomendacao = "" # Limpa o estado
