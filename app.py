# app.py (Versão com Perguntas de Aprofundamento)
import streamlit as st
import os
import re
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
    cultura: str | None = Field(description="A cultura agrícola mencionada, como 'soja' ou 'milho'. Inclui sinônimos como 'plantação', 'lavoura'. Null se não mencionada.")
    praga: str | None = Field(description="A praga mencionada, como 'lagarta', 'guanxuma' ou 'ferrugem'. Inclui termos relacionados como 'doença', 'inseto', 'patógeno'. Null se não mencionada.")
    termo_produto: bool = Field(description="True se a pergunta mencionar termos como 'produto', 'veneno', 'tratamento' ou 'solução'.")


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
    template_transformacao = "Você é um especialista em agronomia. Transforme a pergunta do utilizador em 3 consultas de busca concisas e variadas para uma base de dados vetorial. Responda apenas com as consultas, uma por linha.\n\nPergunta Original: {pergunta}\n\nConsultas de Busca:"
    prompt_transformacao = ChatPromptTemplate.from_template(template_transformacao)
    cadeia_transformacao = prompt_transformacao | llm | StrOutputParser()
    consultas_geradas = cadeia_transformacao.invoke({"pergunta": query}).strip().split('\n')
    
    todos_chunks = []
    for consulta in consultas_geradas:
        todos_chunks.extend(db.similarity_search(consulta, k=4))
    
    unique_chunks = {doc.page_content: doc for doc in todos_chunks}.values()
    if not unique_chunks: return "NAO_ENCONTRADO"

    contexto_final = "\n\n---\n\n".join([doc.page_content for doc in unique_chunks])
    
    aviso_prompt = f"{aviso_contexto}\n\n" if aviso_contexto else ""

    # INSTRUÇÕES REVISTAS (mais flexíveis para informações parciais)
    prompt_geracao_final = f"""{aviso_prompt}Você é um consultor especialista da Agrofel. Com base nos TRECHOS RELEVANTES DAS BULAS, gere uma recomendação clara e objetiva.

PERGUNTA ORIGINAL: "{query}"

TRECHOS RELEVANTES:
---
{contexto_final}
---

INSTRUÇÕES:
1. Sugira produtos mesmo quando faltar cultura OU praga específica
2. Para cada produto:
   - Extraia o NOME EXATO
   - Descreva brevemente sua aplicação
   - Mencione pragas/culturas relevantes dos trechos
3. Se faltar informação:
   - Para cultura não especificada: mencione "Verifique registro para sua cultura"
   - Para praga genérica: sugira para pragas comuns
4. Formato:
   **Produto 1:** [Nome]
   **Aplicação:** [Descrição curta]
   **Indicações:** [Pragas/Culturas relevantes]

   **Produto 2:** [Nome]
   **Aplicação:** [Descrição curta]
   **Indicações:** [Pragas/Culturas relevantes]

5. Se não encontrar produtos adequados, responda APENAS com: "NAO_ENCONTRADO"
6. NUNCA invente nomes de produtos ou informações técnicas"""

    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content

def buscar_resposta_especifica(produto: str, pergunta: str, db, llm):
    """
    Busca informações específicas sobre um produto usando RAG aprimorado.
    """
    # Consulta combinando produto e pergunta
    query = f"{produto} {pergunta}"
    
    # Busca por similaridade com foco no produto
    docs = db.similarity_search(query, k=5, filter={"source": produto})
    if not docs:
        # Fallback: busca geral se não encontrar com filtro
        docs = db.similarity_search(query, k=5)
    
    contexto = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # Prompt especializado para perguntas técnicas
    prompt = f"""
Você é um assistente técnico da Agrofel. Responda à pergunta do usuário baseando-se APENAS nas informações extraídas das bulas de produtos.

PRODUTO EM FOCO: {produto}
PERGUNTA: {pergunta}

INFORMAÇÕES EXTRAÍDAS:
---
{contexto}
---

INSTRUÇÕES:
1. Seja técnico e preciso
2. Se a pergunta for sobre dosagem:
   - Forneça a fórmula de cálculo
   - Dê exemplo para área específica
3. Se não encontrar informação exata:
   - Indique "Informação não disponível"
   - Sugira consultar engenheiro agrônomo
4. Formate a resposta de forma clara e organizada
5. NUNCA invente valores ou especificações técnicas
"""
    resposta = llm.invoke(prompt)
    return resposta.content

def obter_resposta_assistente(query: str, db, llm):
    """
    Orquestra o fluxo de trabalho com Roteador Melhorado
    """
    # ETAPA 1: Análise Semântica
    llm_com_ferramenta = llm.bind_tools([AnalisePergunta])
    analise = llm_com_ferramenta.invoke(query)
    
    # Fallback direto se análise falhar
    if not analise.tool_calls:
        return _buscar_e_gerar_recomendacao(
            query, db, llm, 
            aviso_contexto="🔍 Buscando soluções para sua consulta..."
        )

    dados_analise = analise.tool_calls[0]['args']
    cultura = dados_analise.get("cultura")
    praga = dados_analise.get("praga")
    termo_produto = dados_analise.get("termo_produto", False)

    # ETAPA 2: Roteamento Inteligente
    aviso = None
    
    # Caso 1: Pergunta muito vazia
    if not praga and not cultura and not termo_produto:
        return "Por favor, descreva seu problema agrícola. Ex: 'Preciso de produto para lagarta na soja' ou 'O que usar para ferrugem?'"
    
    # Caso 2: Menções parciais com termo de produto
    if termo_produto:
        if praga and not cultura:
            aviso = f"🔍 Buscando produtos para '{praga}'. Verifique o registro para sua cultura."
        elif cultura and not praga:
            aviso = f"🔍 Buscando produtos recomendados para '{cultura}'. Especifique a praga para recomendações mais precisas."
        elif not praga and not cultura:
            aviso = "🔍 Buscando produtos agrícolas gerais. Especifique cultura/praga para recomendações direcionadas."
    
    # Caso 3: Temos pelo menos um elemento relevante
    return _buscar_e_gerar_recomendacao(
        query, db, llm, 
        aviso_contexto=aviso
    )

# --- Funções de Notificação (mantidas) ---
def enviar_email_confirmacao(pergunta, recomendacao):
    try:
        email_vendedor, email_remetente, senha_remetente = st.secrets["EMAIL_VENDEDOR"], st.secrets["EMAIL_REMETENTE"], st.secrets["SENHA_REMETENTE"]
        corpo_email = f'<html><body><p>Olá,</p><p>Um cliente solicitou um pedido através do <b>Assistente de Campo Agrofel</b>.</p><hr><h3>Detalhes da Solicitação:</h3><p><b>Pergunta do Cliente:</b><br>{pergunta}</p><p><b>Produtos Sugeridos e Confirmados:</b></p><div style="background-color:#f0f0f0; border-left: 5px solid #4CAF50; padding: 10px;">{recomendacao.replace("**", "<b>").replace(chr(10), "<br>")}</div><br><p>Por favor, entre em contato com o cliente para dar seguimento.</p><p>Atenciosamente,<br>Assistente de Campo Agrofel</p></body></html>'
        msg = EmailMessage()
        msg['Subject'], msg['From'], msg['To'] = "Novo Pedido de Cliente - Assistente de Campo Agrofel", email_remetente, email_vendedor
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
    texto_base = f"""Olá! Usei o Assistente de Campo Agrofel e gostaria de falar com um especialista.\n\nMinha pergunta foi: "{pergunta}"\n\nA recomendação foi:\n{recomendacao.replace('**', '')}\n\nAguardo contato."""
    texto_formatado = quote(texto_base)
    return f"https://wa.me/{numero_whatsapp}?text={texto_formatado}"

# --- FUNÇÕES AUXILIARES PARA A INTERFACE ---
def extrair_nomes_produtos(recomendacao: str):
    """ Extrai os nomes dos produtos de uma string de recomendação. """
    padrao = r"\*\*Produto \d+:\*\* (.*?)\n"
    produtos = re.findall(padrao, recomendacao)
    return produtos

# --- INTERFACE DO USUÁRIO ---
st.title("🌿 Assistente de Campo Agrofel")
st.markdown("Bem-vindo! Descreva seu problema com pragas na lavoura e encontrarei a melhor solução para você.")

db, llm = carregar_base_conhecimento()

# Inicialização de estados da sessão
if 'recomendacao' not in st.session_state:
    st.session_state.recomendacao = ""
if 'pergunta' not in st.session_state:
    st.session_state.pergunta = ""
if 'produtos' not in st.session_state:
    st.session_state.produtos = []
if 'resposta_especifica' not in st.session_state:
    st.session_state.resposta_especifica = ""
if 'produto_selecionado' not in st.session_state:
    st.session_state.produto_selecionado = None
if 'pergunta_especifica' not in st.session_state:
    st.session_state.pergunta_especifica = ""

# Formulário principal
with st.form("pergunta_form"):
    pergunta_usuario = st.text_area("Qual praga está afetando sua lavoura e em qual cultura?", height=100, placeholder="Ex: Guanxuma na soja ou 'Produto para lagarta'")
    submitted = st.form_submit_button("Buscar Sugestões")

if submitted and pergunta_usuario:
    if db is not None:
        with st.spinner("Analisando as melhores soluções para você..."):
            st.session_state.pergunta = pergunta_usuario
            recomendacao_gerada = obter_resposta_assistente(pergunta_usuario, db, llm)
            st.session_state.recomendacao = recomendacao_gerada
            # Extrai os nomes dos produtos para perguntas posteriores
            st.session_state.produtos = extrair_nomes_produtos(recomendacao_gerada) if "NAO_ENCONTRADO" not in recomendacao_gerada else []
    else:
        st.error("A base de conhecimento não pôde ser carregada.")

# Exibição da recomendação principal
if st.session_state.recomendacao:
    if "NAO_ENCONTRADO" in st.session_state.recomendacao:
        st.warning("Não encontrei produtos específicos para sua solicitação em nossa base de dados.")
    else:
        st.subheader("Encontrei estas soluções para você:")
        st.markdown(st.session_state.recomendacao)
    
    # Lógica dos botões de ação
    st.markdown("---")
    if "NAO_ENCONTRADO" in st.session_state.recomendacao:
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

        # Seção para perguntas específicas sobre os produtos recomendados
        st.markdown("---")
        st.subheader("Dúvidas técnicas sobre os produtos?")
        st.markdown("Selecione um produto e faça sua pergunta para obter informações detalhadas.")
        
        # Seletor de produtos
        produto_selecionado = st.selectbox(
            "Selecione um produto:",
            st.session_state.produtos,
            index=0,
            key="produto_selector"
        )
        
        # Perguntas frequentes pré-definidas
        perguntas_frequentes = [
            "Como faço pra aplicar esse produto?",
            "Qual quantidade precisa pra 1 hectare?",
            "Esse produto afeta a lavoura?",
            "Quanto tempo após a aplicação posso fazer a colheita?",
            "Qual o modo de ação do produto?",
            "Outra pergunta..."
        ]
        
        pergunta_rapida = st.selectbox(
            "Perguntas frequentes:",
            perguntas_frequentes,
            index=0,
            key="pergunta_rapida"
        )
        
        # Campo para perguntas personalizadas
        if pergunta_rapida == "Outra pergunta...":
            pergunta_personalizada = st.text_input("Faça sua pergunta personalizada:", placeholder="Ex: Posso misturar com outros produtos?")
        else:
            pergunta_personalizada = pergunta_rapida
        
        # Botão para enviar pergunta específica
        if st.button("Obter Resposta Técnica", type="secondary"):
            if produto_selecionado and pergunta_personalizada:
                with st.spinner("Buscando informações técnicas..."):
                    st.session_state.produto_selecionado = produto_selecionado
                    st.session_state.pergunta_especifica = pergunta_personalizada
                    st.session_state.resposta_especifica = buscar_resposta_especifica(
                        produto_selecionado, 
                        pergunta_personalizada, 
                        db, 
                        llm
                    )
            else:
                st.warning("Selecione um produto e faça uma pergunta")

        # Exibir resposta da pergunta específica
        if st.session_state.resposta_especifica:
            st.markdown("---")
            st.subheader(f"Resposta sobre {st.session_state.produto_selecionado}:")
            st.info(st.session_state.resposta_especifica)
            
            # Botão para nova pergunta sobre o mesmo produto
            if st.button("Fazer outra pergunta sobre este produto", type="primary"):
                st.session_state.resposta_especifica = ""
                st.session_state.pergunta_especifica = ""
                st.experimental_rerun()