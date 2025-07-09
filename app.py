# app.py (Versão Ajustada com fallback aprimorado)
import streamlit as st
import os
import smtplib
from email.message import EmailMessage
from urllib.parse import quote

import google.generativeai as genai
from dotenv import load_dotenv

# Imports para LangChain e Pydantic
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

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

# --- DEFINIÇÃO DA FERRAMENTA PARA ANÁLISE SEMÂNTICA ---
class AnalisePergunta(BaseModel):
    """Schema para analisar a pergunta de um agricultor."""
    cultura: str | None = Field(description="A cultura agrícola mencionada, como 'soja' ou 'milho'. Null se não mencionada.")
    praga: str | None = Field(description="A praga mencionada, como 'lagarta', 'guanxuma' ou 'ferrugem'. Null se não mencionada.")
    is_praga_generica: bool = Field(description="True se a praga for uma palavra geral como 'praga', 'doença', 'problema'. False caso contrário.")

# --- FUNÇÕES DE LÓGICA (BACKEND DA APLICAÇÃO) ---
@st.cache_resource(show_spinner="A carregar base de conhecimento...")
def carregar_base_conhecimento():
    """ Carrega o índice FAISS pré-construído do repositório. """
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


def _buscar_e_gerar_recomendacao(query: str, db, llm, aviso_contexto: str | None = None):
    """ Executa o fluxo de RAG com Transformação de Consulta. """
    template_transformacao = (
        "Você é um especialista em agronomia. Transforme a pergunta do utilizador em 3 consultas de busca concisas e variadas para uma base de dados vetorial. "
        "Responda apenas com as consultas, uma por linha.\n\nPergunta Original: {pergunta}\n\nConsultas de Busca:"
    )
    prompt_transformacao = ChatPromptTemplate.from_template(template_transformacao)
    cadeia_transformacao = prompt_transformacao | llm | StrOutputParser()
    consultas_geradas = cadeia_transformacao.invoke({"pergunta": query}).strip().split('\n')
    
    todos_chunks = []
    for consulta in consultas_geradas:
        todos_chunks.extend(db.similarity_search(consulta, k=4))  # Aumentamos para k=4 para mais contexto
    
    unique_chunks = {doc.page_content: doc for doc in todos_chunks}.values()
    if not unique_chunks:
        return "NAO_ENCONTRADO"

    contexto_final = "\n\n---\n\n".join([doc.page_content for doc in unique_chunks])
    
    aviso_prompt = aviso_contexto + "\n\n" if aviso_contexto else ""

    prompt_geracao_final = (
        f"{aviso_prompt}Você é um consultor especialista da Agrofel. Com base nos TRECHOS RELEVANTES DAS BULAS, gere uma recomendação clara e objetiva."
        f"\n\nPERGUNTA ORIGINAL DO AGRICULTOR: \"{query}\"\n\nTRECHOS RELEVANTES DAS BULAS:\n---\n{contexto_final}\n---\nINSTRUÇÕES:\n"
        "1. Esforce-se para sugerir DOIS produtos distintos. Se for impossível, sugira apenas um.\n"
        "2. Para cada produto, extraia o nome exato e crie uma descrição curta e convincente.\n"
        "3. Formato:\n   **Produto 1:** [Nome do Produto]\n   **Descrição:** [Sua descrição]\n\n"
        "   **Produto 2:** [Nome do Produto]\n   **Descrição:** [Sua descrição]\n"
        "4. Responda *apenas* no formato especificado. Não inclua nenhuma outra informação, explicação ou comentário extra.\n"
        "5. Se os trechos não forem suficientes para nenhuma recomendação segura, responda APENAS com: \"NAO_ENCONTRADO\"."
    )
    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content


def obter_resposta_assistente(query: str, db, llm):
    """
    Orquestra o fluxo de trabalho com o Roteador Flexível Aprimorado.
    """
    # ETAPA 1: Análise Semântica
    llm_com_ferramenta = llm.bind_tools([AnalisePergunta])
    analise = llm_com_ferramenta.invoke(query)
    
    if not analise.tool_calls:
        aviso = (
            "Não consegui analisar a sua pergunta em detalhe, mas aqui estão alguns resultados baseados nas palavras-chave. "
            "Para uma resposta mais precisa, por favor, especifique a praga e a cultura."
        )
        return _buscar_e_gerar_recomendacao(query, db, llm, aviso_contexto=aviso)

    dados_analise = analise.tool_calls[0]['args']
    cultura = dados_analise.get("cultura")
    praga = dados_analise.get("praga")
    is_praga_generica = dados_analise.get("is_praga_generica")

    # ETAPA 2: Roteamento Flexível Aprimorado
    if not praga:
        aviso = (
            "Não foi possível identificar a praga mencionada, porém apresento recomendações gerais com base na sua pergunta."
        )
        return _buscar_e_gerar_recomendacao(query, db, llm, aviso_contexto=aviso)

    if is_praga_generica:
        aviso = (
            f"Sua pergunta sobre '{praga}' em '{cultura or 'sua lavoura'}' é um pouco genérica, "
            "entretanto seguem recomendações baseadas nos termos fornecidos."
        )
        return _buscar_e_gerar_recomendacao(query, db, llm, aviso_contexto=aviso)

    if not cultura:
        aviso = (
            "Como a sua pergunta não especificou uma cultura, as recomendações abaixo são gerais. "
            "Verifique sempre se o produto é registado para a sua cultura específica."
        )
        return _buscar_e_gerar_recomendacao(query, db, llm, aviso_contexto=aviso)

    # Pergunta completa e específica: busca normal
    return _buscar_e_gerar_recomendacao(query, db, llm)

# --- Funções de Notificação (sem alterações) ---
def enviar_email_confirmacao(pergunta, recomendacao):
    try:
        email_vendedor, email_remetente, senha_remetente = (
            st.secrets["EMAIL_VENDEDOR"], st.secrets["EMAIL_REMETENTE"], st.secrets["SENHA_REMETENTE"]
        )
        corpo_email = (
            f'<html><body><p>Olá,</p><p>Um cliente solicitou um pedido através do <b>Assistente de Campo Agrofel</b>.</p>'
            '<hr><h3>Detalhes da Solicitação:</h3>'
            f'<p><b>Pergunta do Cliente:</b><br>{pergunta}</p>'
            '<p><b>Produtos Sugeridos e Confirmados:</b></p>'
            '<div style="background-color:#f0f0f0; border-left: 5px solid #4CAF50; padding: 10px;">'
            f'{recomendacao.replace("**", "<b>").replace(chr(10), "<br>")}</div><br>'
            '<p>Por favor, entre em contato com o cliente para dar seguimento.</p>'
            '<p>Atenciosamente,<br>Assistente de Campo Agrofel</p>'
            '</body></html>'
        )
        msg = EmailMessage()
        msg['Subject'], msg['From'], msg['To'] = (
            "Novo Pedido de Cliente - Assistente de Campo Agrofel",
            email_remetente,
            email_vendedor
        )
        msg.add_alternative(corpo_email, subtype='html')
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_remetente, senha_remetente)
            smtp.send_message(msg)
        st.success("Pedido confirmado! Seu consultor Agrofel foi notificado por email.")
        st.balloons()
    except Exception as e:
        st.error(f"Ocorreu um erro ao enviar o email: {e}")


def gerar_link_whatsapp(pergunta, recomendacao):
    numero_whatsapp = "5519989963385"
    texto_base = (
        f"Olá! Usei o Assistente de Campo Agrofel e gostaria de falar com um especialista.\n\n"
        f"Minha pergunta foi: \"{pergunta}\"\n\nA recomendação foi:\n{recomendacao.replace('**', '')}\n\nAguardo contato."
    )
    return f"https://wa.me/{numero_whatsapp}?text={quote(texto_base)}"

# --- INTERFACE DO USUÁRIO ---
st.title("🌿 Assistente de Campo Agrofel")
st.markdown("Bem-vindo! Descreva seu problema com pragas na lavoura e encontrarei a melhor solução para você.")

db, llm = carregar_base_conhecimento()

if 'recomendacao' not in st.session_state:
    st.session_state.recomendacao = ""
if 'pergunta' not in st.session_state:
    st.session_state.pergunta = ""

with st.form("pergunta_form"):
    pergunta_usuario = st.text_area(
        "Qual praga está afetando sua lavoura e em qual cultura?",
        height=100,
        placeholder="Ex: Guanxuma na soja"
    )
    submitted = st.form_submit_button("Buscar Sugestões")

if submitted and pergunta_usuario:
    if db is not None:
        with st.spinner("Analisando as melhores soluções para você..."):
            st.session_state.pergunta = pergunta_usuario
            recomendacao_gerada = obter_resposta_assistente(pergunta_usuario, db, llm)
            st.session_state.recomendacao = recomendacao_gerada
    else:
        st.error("A base de conhecimento não pôde ser carregada.")

if st.session_state.recomendacao:
    if "NAO_ENCONTRADO" in st.session_state.recomendacao:
        st.warning("Não encontrei produtos específicos para sua solicitação em nossa base de dados.")
    elif any(x in st.session_state.recomendacao for x in ["Para que eu possa ajudar", "A sua pergunta"]) :
        st.info(st.session_state.recomendacao)
    else:
        st.subheader("Encontrei estas sugestões para você:")
        st.markdown(st.session_state.recomendacao)
    
    st.markdown("---")
    if any(x in st.session_state.recomendacao for x in ["NAO_ENCONTRADO","Para que eu possa ajudar","A sua pergunta"]):
        st.info("Gostaria de falar com um de nossos consultores para obter ajuda personalizada?")
        link_whatsapp_sem_produto = gerar_link_whatsapp(st.session_state.pergunta, "Nenhuma recomendação automática foi gerada.")
        st.link_button("🗣️ Falar com um Humano via WhatsApp", link_whatsapp_sem_produto, use_container_width=True)
    else:
        st.markdown("O que você gostaria de fazer?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirmar Pedido", type="primary", use_container_width=True):
                enviar_email_confirmacao(st.session_state.pergunta, st.session_state.recomendacao)
                st.session_state.recomendacao = ""
        with col2:
            link_whatsapp_com_produto = gerar_link_whatsapp(st.session_state.pergunta, st.session_state.recomendacao)
            st.link_button("🗣️ Falar com um Humano via WhatsApp", link_whatsapp_com_produto, use_container_width=True)
