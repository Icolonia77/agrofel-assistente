# 2_Testar_Base.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Chave de API do Google não encontrada! Verifique seu arquivo .env")
genai.configure(api_key=api_key)

# Caminho para o índice FAISS
CAMINHO_INDEX_FAISS = "C:/Users/Usuario/Documents/Agrofel/faiss_index_agrofel"

def diagnosticar_busca(query: str):
    """
    Executa o processo de busca passo a passo para diagnóstico.
    """
    print("="*50)
    print("INICIANDO DIAGNÓSTICO")
    print(f"Consulta (Query): '{query}'")
    print("="*50)

    # --- Carregando a Base e o Modelo ---
    if not os.path.exists(CAMINHO_INDEX_FAISS):
        print(f"ERRO: Índice FAISS não encontrado em '{CAMINHO_INDEX_FAISS}'")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(CAMINHO_INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
        print("Base de conhecimento e LLM carregados com sucesso.\n")
    except Exception as e:
        print(f"ERRO ao carregar a base de dados: {e}")
        return

    # --- ETAPA 1: TESTE DE RECUPERAÇÃO (RETRIEVAL) ---
    print("\n--- ETAPA 1: Testando a Busca por Similaridade ---")
    # Aumentamos o 'k' para 5 para ter mais chances de encontrar o chunk relevante
    docs_relevantes = db.similarity_search_with_score(query, k=5)
    
    if not docs_relevantes:
        print("RESULTADO: A busca por similaridade não retornou NENHUM documento. A base pode estar vazia ou a consulta é muito diferente dos dados.")
        return

    print(f"Encontrados {len(docs_relevantes)} chunks de texto relevantes. Analisando:")
    for i, (doc, score) in enumerate(docs_relevantes):
        print(f"\nChunk {i+1} (Score: {score:.4f})")
        print(f"  Fonte: {doc.metadata.get('source', 'N/D')}")
        print(f"  Página: {doc.metadata.get('page', 'N/A')}")
        print(f"  Conteúdo: '{doc.page_content[:400]}...'")
    
    # --- ETAPA 2: TESTE DE GERAÇÃO DE CONTEXTO ---
    print("\n\n--- ETAPA 2: Construindo o Contexto para o LLM ---")
    contexto = "\n\n".join([doc.page_content for doc, score in docs_relevantes])
    print("O seguinte contexto será enviado para o modelo de linguagem:")
    print("-" * 40)
    print(contexto)
    print("-" * 40)

    # --- ETAPA 3: TESTE DE GERAÇÃO PELO LLM ---
    print("\n\n--- ETAPA 3: Testando a Resposta do LLM com o Contexto Acima ---")
    prompt_template = f"""
    Você é um consultor especialista da Agrofel. Sua tarefa é analisar a pergunta de um agricultor
    e, com base nos trechos de bulas fornecidos, gerar uma recomendação clara e objetiva.

    PERGUNTA DO AGRICULTOR: "{query}"

    TRECHOS RELEVANTES DAS BULAS:
    ---
    {contexto}
    ---

    INSTRUÇÕES:
    1. Com base nos trechos, identifique e sugira os DOIS melhores produtos para a necessidade do agricultor.
    2. Para cada produto, crie uma descrição curta e convincente, destacando seu principal benefício para o problema apresentado.
    3. Apresente a resposta no seguinte formato, e somente neste formato:
       
       Produto 1: [Nome do Produto]
       Descrição: [Sua descrição persuasiva aqui]

       Produto 2: [Nome do Produto]
       Descrição: [Sua descrição persuasiva aqui]
    
    4. Se os trechos não contiverem informação suficiente para uma recomendação clara, responda APENAS com: "NAO_ENCONTRADO".
    """
    
    resposta_final = llm.invoke(prompt_template).content
    print("\nResposta Final Gerada pelo LLM:")
    print(resposta_final)
    print("\n\nFIM DO DIAGNÓSTICO")
    print("="*50)


if __name__ == "__main__":
    # Use a consulta específica que você sabe que deveria funcionar
    consulta_teste = "Qual produto usar para Capim-amargoso na cultura da soja?"
    diagnosticar_busca(consulta_teste)