# app.py (Versão Final com Agente Conversacional, Intenção, Ferramentas e Guardrails)
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
class ResponderConversa(BaseModel):
    """Ferramenta para responder a saudações, despedidas ou conversas informais que não são perguntas técnicas."""
    resposta_cordial: str = Field(description="Uma resposta curta, amigável e profissional. Ex: 'Boa noite! Em que posso ajudar?'.")

class BuscaRecomendacao(BaseModel):
    """Ferramenta para buscar recomendações gerais de produtos para um problema agrícola."""
    problema_agricola: str = Field(description="A descrição do problema do utilizador, incluindo praga e cultura se mencionados. Ex: 'guanxuma na soja'.")

class BuscaTecnica(BaseModel):
    """Ferramenta para buscar uma resposta técnica sobre um produto específico."""
    nome_produto: str = Field(description="O nome do produto sobre o qual se pergunta, extraído do histórico da conversa. Ex: 'GLYPHOTAL TR'.")
    pergunta_tecnica: str = Field(description="A pergunta técnica específica. Ex: 'qual a dosagem para 1 hectare?'.")


# --- FUNÇÕES DE LÓGICA (BACKEND DA APLICAÇÃO) ---

@st.cache_resource(show_spinner="A carregar base de conhecimento...")
def carregar_base_conhecimento():
    """ Carrega o índice FAISS pré-construído do repositório. """
    CAMINHO_INDEX_FAISS = "faiss_index_agrofel"
    if not os.path.exists(CAMINHO_INDEX_FAISS):
        st.error(f"ERRO CRÍTICO: A base de conhecimento ('{CAMINHO_INDEX_FAISS}') não foi encontrada.")
        return None, None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
        return db, llm
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar a base de conhecimento: {e}")
        return None, None

def _run_rag_chain(query: str, db, llm, prompt_template: str):
    """ Função genérica para executar uma cadeia RAG. """
    docs = db.similarity_search(query, k=5)
    if not docs:
        return "Com base nas informações disponíveis, não encontrei uma resposta específica na nossa base de dados. Poderia reformular a sua pergunta?"
    
    contexto = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    cadeia = prompt | llm | StrOutputParser()
    return cadeia.invoke({"contexto": contexto, "pergunta": query})

def ferramenta_buscar_recomendacao(problema_agricola: str, db, llm):
    """ Ferramenta que busca recomendações de produtos. """
    prompt_template = """
Você é um consultor especialista da Agrofel. Sua tarefa é gerar uma recomendação de produtos com base na pergunta do cliente e nas informações das bulas.
Seja cordial e profissional.

PERGUNTA DO CLIENTE: "{pergunta}"

INFORMAÇÕES RELEVANTES DAS BULAS:
---
{contexto}
---

INSTRUÇÕES:
1. Com base **estritamente** nas informações fornecidas, sugira até DOIS produtos relevantes.
2. Para cada produto, extraia o NOME EXATO e crie uma descrição curta e convincente.
3. Formato da Resposta:
   **Produto 1:** [Nome do Produto]
   **Descrição:** [Sua descrição]

   **Produto 2:** [Nome do Produto]
   **Descrição:** [Sua descrição]
4. Se as informações não forem suficientes para uma recomendação segura, responda APENAS com: "Com base nas informações disponíveis, não encontrei um produto específico para a sua solicitação. Poderia reformular a sua pergunta ou gostaria de falar com um especialista?"
5. NUNCA invente nomes de produtos ou informações técnicas.
"""
    return _run_rag_chain(problema_agricola, db, llm, prompt_template)

def ferramenta_buscar_resposta_tecnica(nome_produto: str, pergunta_tecnica: str, db, llm):
    """ Ferramenta que busca respostas técnicas sobre um produto. """
    query = f"informações sobre {nome_produto} para responder: {pergunta_tecnica}"
    prompt_template = """
Você é um assistente técnico da Agrofel. Sua tarefa é responder a uma pergunta técnica sobre um produto, baseando-se **exclusivamente** nas informações das bulas.
Seja preciso e cordial.

PRODUTO: "{pergunta}"
INFORMAÇÕES RELEVANTES DAS BULAS:
---
{contexto}
---

INSTRUÇÕES:
1. Responda à pergunta do utilizador de forma clara e direta, usando apenas os dados fornecidos.
2. Se a informação exata não estiver nos trechos, responda: "Não encontrei esta informação específica na bula do produto. Para detalhes técnicos, recomendo consultar um engenheiro agrônomo ou falar com um de nossos especialistas."
3. NUNCA invente valores, dosagens ou especificações.
"""
    return _run_rag_chain(query, db, llm, prompt_template)

def orquestrador_conversacional(query: str, chat_history: list, db, llm):
    """
    O cérebro do agente. Analisa a intenção e chama a ferramenta correta.
    """
    historico_formatado = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    prompt_roteador = f"""
Você é o orquestrador de um chatbot de agronomia. Sua tarefa é analisar a ÚLTIMA PERGUNTA DO UTILIZADOR e o HISTÓRICO DA CONVERSA para decidir qual ferramenta chamar. Seja cordial e profissional.

- Se a pergunta for uma saudação, despedida ou conversa informal (ex: 'Olá', 'Boa noite', 'Obrigado', 'Quem é você?'), use a ferramenta `ResponderConversa`.
- Se a pergunta for geral, sobre um problema agrícola (ex: 'praga na soja', 'o que usar para guanxuma'), use a ferramenta `BuscaRecomendacao`.
- Se a pergunta for claramente sobre um produto específico já mencionado no histórico (ex: 'qual a dosagem desse produto?', 'como aplicar o Glyphotal?'), use a ferramenta `BuscaTecnica`.

HISTÓRICO DA CONVERSA:
{historico_formatado}

ÚLTIMA PERGUNTA DO UTILIZADOR: {query}
"""
    
    llm_com_ferramentas = llm.bind_tools([BuscaRecomendacao, BuscaTecnica, ResponderConversa])
    analise = llm_com_ferramentas.invoke(prompt_roteador)

    if not analise.tool_calls:
        return "Peço desculpa, não consegui entender a sua pergunta. Poderia tentar reformulá-la de outra maneira?"
    
    ferramenta_chamada = analise.tool_calls[0]
    nome_ferramenta = ferramenta_chamada['name']
    argumentos = ferramenta_chamada['args']

    if nome_ferramenta == ResponderConversa.__name__:
        return argumentos.get("resposta_cordial", "Olá! Como posso ajudar?")

    if nome_ferramenta == BuscaTecnica.__name__:
        return ferramenta_buscar_resposta_tecnica(
            nome_produto=argumentos.get("nome_produto"),
            pergunta_tecnica=argumentos.get("pergunta_tecnica"),
            db=db,
            llm=llm
        )
    
    return ferramenta_buscar_recomendacao(
        problema_agricola=argumentos.get("problema_agricola", query),
        db=db,
        llm=llm
    )

# --- Guardrail de Segurança de Entrada ---
def is_input_safe(query: str) -> bool:
    """ Verifica se a entrada do utilizador contém linguagem inadequada. """
    # Esta é uma lista de exemplo. Pode ser expandida com mais termos.
    blocklist = ["palavrão", "insulto", "ofensa", "agressão"]
    query_lower = query.lower()
    if any(word in query_lower for word in blocklist):
        return False
    return True

# --- Funções de Notificação (sem alterações) ---
def enviar_email_confirmacao(pergunta, recomendacao):
    # ... (código de envio de email aqui)
    pass

def gerar_link_whatsapp(pergunta, recomendacao):
    # ... (código do link WhatsApp aqui)
    pass

# --- INTERFACE DO USUÁRIO (CHAT) ---
st.title("🌿 Assistente de Campo Agrofel")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou o assistente virtual da Agrofel. Como posso ajudar com a sua lavoura hoje?"}]

db, llm = carregar_base_conhecimento()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Adicionar botões de ação às mensagens do assistente que contêm recomendações
        if message["role"] == "assistant" and "Produto 1:" in message["content"]:
            st.markdown("---")
            # ... (Lógica dos botões de ação pode ser adicionada aqui se desejado)

if prompt := st.chat_input("Descreva o seu problema ou faça uma pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Execução com Guardrail ---
    if not is_input_safe(prompt):
        response = "Peço desculpa, mas não posso processar pedidos com linguagem inadequada. Por favor, mantenha a conversa profissional e focada em questões agrícolas."
    elif db is not None and llm is not None:
        with st.chat_message("assistant"):
            with st.spinner("A pensar..."):
                historico_para_analise = st.session_state.messages[:-1]
                response = orquestrador_conversacional(prompt, historico_para_analise, db, llm)
                st.markdown(response)
    else:
        response = "Não foi possível conectar à base de conhecimento. Por favor, recarregue a página."
    
    st.session_state.messages.append({"role": "assistant", "content": response})
