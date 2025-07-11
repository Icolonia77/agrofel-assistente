# app.py (Versão Final com Interface Conversacional)
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


# --- DEFINIÇÃO DAS FERRAMENTAS PARA O AGENTE ---
class AnalisePerguntaGeral(BaseModel):
    """Schema para analisar uma pergunta geral sobre um problema agrícola."""
    cultura: str | None = Field(description="A cultura agrícola mencionada, como 'soja' ou 'milho'. Null se não mencionada.")
    praga: str | None = Field(description="A praga mencionada, como 'lagarta' ou 'guanxuma'. Null se não mencionada.")

class AnalisePerguntaTecnica(BaseModel):
    """Schema para analisar uma pergunta técnica sobre um produto específico."""
    nome_produto: str = Field(description="O nome exato do produto sobre o qual se pergunta.")
    pergunta_tecnica: str = Field(description="A pergunta específica sobre o produto, como 'qual a dosagem' ou 'modo de aplicação'.")


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

def buscar_recomendacao_produto(query: str, db, llm):
    """ Executa o fluxo de RAG para encontrar produtos. """
    template_transformacao = "Transforme a pergunta do utilizador em 3 consultas de busca concisas e variadas para uma base de dados vetorial. Responda apenas com as consultas, uma por linha.\n\nPergunta Original: {pergunta}"
    prompt_transformacao = ChatPromptTemplate.from_template(template_transformacao)
    cadeia_transformacao = prompt_transformacao | llm | StrOutputParser()
    consultas_geradas = cadeia_transformacao.invoke({"pergunta": query}).strip().split('\n')
    
    todos_chunks = []
    for consulta in consultas_geradas:
        todos_chunks.extend(db.similarity_search(consulta, k=4))
    
    unique_chunks = {doc.page_content: doc for doc in todos_chunks}.values()
    if not unique_chunks: return "NAO_ENCONTRADO"

    contexto_final = "\n\n---\n\n".join([doc.page_content for doc in unique_chunks])
    
    prompt_geracao_final = f"""Você é um consultor especialista da Agrofel. Com base nos TRECHOS RELEVANTES DAS BULAS, gere uma recomendação clara.\n\nPERGUNTA ORIGINAL: "{query}"\n\nTRECHOS RELEVANTES:\n---\n{contexto_final}\n---\nINSTRUÇÕES:\n1. Sugira até DOIS produtos distintos.\n2. Formato:\n   **Produto 1:** [Nome]\n   **Descrição:** [Descrição curta e convincente]\n\n   **Produto 2:** [Nome]\n   **Descrição:** [Descrição curta e convincente]\n3. Se não encontrar produtos, responda APENAS com: "NAO_ENCONTRADO"."""
    resposta_final = llm.invoke(prompt_geracao_final)
    return resposta_final.content

def buscar_resposta_tecnica(produto: str, pergunta: str, db, llm):
    """ Busca informações técnicas específicas sobre um produto. """
    query = f"informações sobre {produto}: {pergunta}"
    docs = db.similarity_search(query, k=5)
    contexto = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Você é um assistente técnico da Agrofel. Responda à pergunta do usuário baseando-se APENAS nas informações das bulas.\n\nPRODUTO: {produto}\nPERGUNTA: {pergunta}\n\nINFORMAÇÕES EXTRAÍDAS:\n---\n{contexto}\n---\nINSTRUÇÕES:\n1. Seja técnico e preciso.\n2. Se não encontrar a informação, indique "Informação não disponível na bula." e sugira consultar um engenheiro agrônomo."""
    resposta = llm.invoke(prompt)
    return resposta.content

def orquestrador_conversacional(query: str, chat_history: list, db, llm):
    """
    O cérebro do agente. Decide qual ação tomar com base na pergunta do usuário e no histórico.
    """
    # Adiciona o histórico ao prompt para dar contexto ao LLM
    historico_formatado = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    prompt_roteador = f"""
Você é o orquestrador de um chatbot de agronomia. Analise a ÚLTIMA PERGUNTA DO UTILIZADOR à luz do HISTÓRICO DA CONVERSA.
Decida qual ferramenta usar: `buscar_recomendacao_produto` ou `buscar_resposta_tecnica`.

- Use `buscar_resposta_tecnica` se a pergunta for sobre um produto específico já mencionado no histórico.
- Caso contrário, use `buscar_recomendacao_produto`.

HISTÓRICO DA CONVERSA:
{historico_formatado}

ÚLTIMA PERGUNTA DO UTILIZADOR: {query}
"""
    
    # Define as ferramentas que o LLM pode usar
    llm_com_ferramentas = llm.bind_tools([AnalisePerguntaGeral, AnalisePerguntaTecnica])
    
    # O LLM decide qual ferramenta chamar
    analise = llm_com_ferramentas.invoke(prompt_roteador)

    if not analise.tool_calls:
        return "Não consegui entender a sua pergunta. Poderia reformulá-la?"
    
    # Executa a ferramenta escolhida pelo LLM
    ferramenta_chamada = analise.tool_calls[0]
    nome_ferramenta = ferramenta_chamada['name']
    argumentos = ferramenta_chamada['args']

    if nome_ferramenta == AnalisePerguntaTecnica.__name__:
        produto = argumentos.get("nome_produto")
        pergunta_tecnica = argumentos.get("pergunta_tecnica")
        if produto and pergunta_tecnica:
            return buscar_resposta_tecnica(produto, pergunta_tecnica, db, llm)
    
    # Por defeito, ou se a ferramenta for AnalisePerguntaGeral, busca uma recomendação de produto.
    return buscar_recomendacao_produto(query, db, llm)


# --- Funções de Notificação (sem alterações) ---
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


# --- INTERFACE DO USUÁRIO (CHAT) ---
st.title("🌿 Assistente de Campo Agrofel")
st.markdown("---")

# Inicialização do estado da sessão para o chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou o assistente virtual da Agrofel. Como posso ajudar com a sua lavoura hoje?"}]

# Carregar a base de conhecimento e o LLM
db, llm = carregar_base_conhecimento()

# Exibir mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Adicionar botões de ação às mensagens do assistente que contêm recomendações
        if message["role"] == "assistant" and "Produto 1:" in message["content"]:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                # O uso de uma chave única para o botão é crucial
                if st.button("✅ Confirmar Pedido", key=f"email_{len(st.session_state.messages)}", use_container_width=True):
                    enviar_email_confirmacao(st.session_state.messages[-2]['content'], message['content'])
            with col2:
                link_whatsapp = gerar_link_whatsapp(st.session_state.messages[-2]['content'], message['content'])
                st.link_button("🗣️ Falar com um Humano", link_whatsapp, use_container_width=True)

# Input do utilizador
if prompt := st.chat_input("Descreva o seu problema ou faça uma pergunta..."):
    # Adicionar mensagem do utilizador ao histórico e à tela
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gerar e exibir a resposta do assistente
    if db is not None and llm is not None:
        with st.chat_message("assistant"):
            with st.spinner("A pensar..."):
                # Passa o histórico da conversa para o orquestrador
                historico_para_analise = st.session_state.messages[:-1] # Exclui a pergunta atual
                response = orquestrador_conversacional(prompt, historico_para_analise, db, llm)
                
                # Armazena a resposta antes de exibir
                st.session_state.response_holder = response

        # Adiciona a resposta do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.response_holder})
        # Força o recarregamento da página para exibir a nova mensagem com os botões
        st.rerun()
    else:
        st.error("Não foi possível conectar à base de conhecimento. Por favor, recarregue a página.")
