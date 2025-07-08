# app.py (Versão Final de Produção com Notificações)
import streamlit as st
import os
import smtplib
from email.message import EmailMessage
from urllib.parse import quote

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

    template_transformacao = """Você é um especialista em agronomia. Sua tarefa é transformar a pergunta de um utilizador numa lista de 3 consultas de busca otimizadas para uma base de dados vetorial.
    As consultas devem ser concisas e variadas para cobrir diferentes aspetos da pergunta.
    Responda apenas com as consultas, uma por linha.

    Pergunta Original: {pergunta}

    Consultas de Busca:
    """
    prompt_transformacao = ChatPromptTemplate.from_template(template_transformacao)
    
    cadeia_transformacao = prompt_transformacao | llm | StrOutputParser()
    consultas_geradas = cadeia_transformacao.invoke({"pergunta": query}).strip().split('\n')

    todos_chunks = []
    for consulta in consultas_geradas:
        todos_chunks.extend(db.similarity_search(consulta, k=3))
    
    unique_chunks = {doc.page_content: doc for doc in todos_chunks}.values()

    if not unique_chunks:
        return "NAO_ENCONTRADO"

    contexto_final = "\n\n---\n\n".join([doc.page_content for doc in unique_chunks])

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

def enviar_email_confirmacao(pergunta, recomendacao):
    """
    Envia um email de notificação para o vendedor.
    Requer configuração de segredos no Streamlit.
    """
    try:
        # Carrega as credenciais dos segredos do Streamlit
        email_vendedor = st.secrets["EMAIL_VENDEDOR"]
        email_remetente = st.secrets["EMAIL_REMETENTE"]
        senha_remetente = st.secrets["SENHA_REMETENTE"]
    except (FileNotFoundError, KeyError):
        st.error("As credenciais de email não estão configuradas nos segredos do Streamlit. O email não pode ser enviado.")
        return

    # Formata o corpo do email em HTML
    corpo_email = f"""
    <html>
    <body>
        <p>Olá,</p>
        <p>Um cliente solicitou um pedido através do <b>Assistente de Campo Agrofel</b>.</p>
        <hr>
        <h3>Detalhes da Solicitação:</h3>
        <p><b>Pergunta do Cliente:</b><br>{pergunta}</p>
        <p><b>Produtos Sugeridos e Confirmados:</b></p>
        <div style="background-color:#f0f0f0; border-left: 5px solid #4CAF50; padding: 10px;">
            {recomendacao.replace('**', '<b>').replace('**', '</b>').replace(chr(10), '<br>')}
        </div>
        <br>
        <p>Por favor, entre em contato com o cliente para dar seguimento.</p>
        <p>Atenciosamente,<br>Assistente de Campo Agrofel</p>
    </body>
    </html>
    """

    # Cria o objeto do email
    msg = EmailMessage()
    msg['Subject'] = "Novo Pedido de Cliente - Assistente de Campo Agrofel"
    msg['From'] = email_remetente
    msg['To'] = email_vendedor
    msg.set_content("Este é um email em HTML. Por favor, ative a visualização de HTML.")
    msg.add_alternative(corpo_email, subtype='html')

    # Envia o email usando o servidor SMTP do Gmail
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_remetente, senha_remetente)
            smtp.send_message(msg)
        st.success("Pedido confirmado! Seu consultor Agrofel foi notificado por email.")
        st.balloons()
    except Exception as e:
        st.error(f"Ocorreu um erro ao enviar o email: {e}")

def gerar_link_whatsapp(pergunta, recomendacao):
    """
    Gera um link "click to chat" do WhatsApp com uma mensagem pré-formatada.
    """
    numero_whatsapp = "5519989963385" # Formato internacional sem '+' ou espaços
    texto_base = f"""Olá!
Usei o Assistente de Campo Agrofel e gostaria de falar com um especialista.

Minha pergunta foi: "{pergunta}"

A recomendação foi:
{recomendacao.replace('**', '')}

Aguardo contato.
"""
    texto_formatado = quote(texto_base) # Codifica o texto para a URL
    return f"https://wa.me/{numero_whatsapp}?text={texto_formatado}"


# --- INTERFACE DO USUÁRIO ---
st.title("🌿 Assistente de Campo Agrofel")
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
        st.info("Gostaria de falar com um de nossos consultores para obter ajuda personalizada?")
        
        # Gera o link do WhatsApp para o caso de não encontrar produtos
        link_whatsapp_sem_produto = gerar_link_whatsapp(st.session_state.pergunta, "Nenhuma recomendação automática foi gerada.")
        st.link_button("🗣️ Falar com um Humano via WhatsApp", link_whatsapp_sem_produto, use_container_width=True)
    else:
        st.subheader("Encontrei estas sugestões para você:")
        st.markdown(st.session_state.recomendacao)
        
        st.markdown("---")
        st.markdown("O que você gostaria de fazer?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirmar Pedido", type="primary", use_container_width=True):
                enviar_email_confirmacao(st.session_state.pergunta, st.session_state.recomendacao)
                st.session_state.recomendacao = "" # Limpa o estado após a ação
        with col2:
            # Gera o link do WhatsApp com a recomendação incluída
            link_whatsapp_com_produto = gerar_link_whatsapp(st.session_state.pergunta, st.session_state.recomendacao)
            st.link_button("🗣️ Falar com um Humano via WhatsApp", link_whatsapp_com_produto, use_container_width=True)


