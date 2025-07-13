import os
from dotenv import load_dotenv  
import numpy as np
import pandas as pd
import faiss
import re
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from openai import OpenAI
import streamlit as st


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)  

@st.cache_data
def download_nltk_resources():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")

download_nltk_resources()

########################################################################
# LECTURA DATA
########################################################################

EXCEL_PATH =  "Chunks3.xlsx"
EMBEDDINGS_FILE =  "embeddings.npy"

@st.cache_data
def load_data():
    df = pd.read_excel(EXCEL_PATH)
    return df

data = load_data()

########################################################################
# LIMPIEZA MÍNIMA
########################################################################

def minimal_clean(text):
    text = text.replace("\n", " ")
    return text.strip()

texts = [minimal_clean(t) for t in data["Texto"].tolist()]

########################################################################
# DETECTAR PRODUCTO EN QUERY
########################################################################

def detectar_producto_en_query(query):
    """
    Retorna el primer 'EMPXXX' encontrado, o None si no encuentra.
    """
    pattern = r"\b(EMP\d+)\b"
    match = re.search(pattern, query.upper())
    if match:
        return match.group(1)  
    return None

########################################################################
# EMBEDDINGS: CARGA + NORMALIZACIÓN
########################################################################

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def normalize_embeddings(embs):
    return np.array([normalize_vector(e) for e in embs], dtype=np.float32)

@st.cache_data
def load_embeddings():
    """
    Carga embeddings desde archivo y los normaliza (para coseno).
    Asume que el orden de los embeddings corresponde a data.index
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error("No se encontró el archivo de embeddings. Genera embeddings primero.")
    embeddings = np.load(EMBEDDINGS_FILE)
    embeddings = normalize_embeddings(embeddings)
    return embeddings

embeddings_full = load_embeddings()

########################################################################
# FUNCIÓN DE FILTRADO POR PRODUCTO
########################################################################

def filtrar_por_producto(df, embeddings, producto):
    """
    Filtra el dataframe y los embeddings para quedarnos
    solo con las filas donde aparezca 'producto' en la columna 'Productos'.
    """
    mask = df["Productos"].astype(str).str.upper().str.contains(producto.upper(), na=False)
    df_filtrado = df[mask].copy()
    embeddings_filtrados = embeddings[mask.values, :] 
    return df_filtrado, embeddings_filtrados

########################################################################
# (Opcional) Construcción de índice FAISS
########################################################################

def build_faiss_index(emb_array):
    index_temp = faiss.IndexFlatIP(emb_array.shape[1])
    index_temp.add(emb_array)
    return index_temp

########################################################################
# Reranking
########################################################################

def local_rerank(query_emb, top_texts, top_indices, top_distances, embeddings_subset):
    """
    Re-ranqueo por dot product usando el subset de embeddings.
    """
    to_rank = []
    for i, dist in enumerate(top_distances):
        chunk_emb = embeddings_subset[top_indices[i]]
        dot_product = np.dot(query_emb, chunk_emb)
        to_rank.append((top_texts[i], top_indices[i], dist, dot_product))

    re_ranked = sorted(to_rank, key=lambda x: x[3], reverse=True)
    return re_ranked

########################################################################
# Búsqueda con prefiltro por producto
########################################################################

@st.cache_data
def search_similar(query, k=10):
    # 1) Detectar producto
    producto = detectar_producto_en_query(query)

    # 2) Filtrar data + embeddings (si corresponde)
    if producto:
        df_sub, emb_sub = filtrar_por_producto(data, embeddings_full, producto)
        if len(df_sub) == 0:
            df_sub = data
            emb_sub = embeddings_full
    else:
        df_sub = data
        emb_sub = embeddings_full

    # 3) FAISS
    index_sub = build_faiss_index(emb_sub)

    # 4) Embedding query
    query_clean = minimal_clean(query)
    response = client.embeddings.create(
        input=query_clean,
        model="text-embedding-3-large"
    )
    query_emb = np.array(response.data[0].embedding, dtype=np.float32)
    query_emb = normalize_vector(query_emb).reshape(1, -1)

    # 5) Buscamos en índice
    distances, indices = index_sub.search(query_emb, k)

    # 6) Extraemos textos y metadatos
    df_sub_reset = df_sub.reset_index(drop=True)
    results_text = []
    results_circ = []
    results_fecha = []

    for i in indices[0]:
        row = df_sub_reset.iloc[i]
        results_text.append(row["Texto"])
        results_circ.append(row.get("Circular", "N/A"))
        results_fecha.append(row.get("Fecha", "N/A"))

    # 7) Reranking
    re_ranked = local_rerank(query_emb[0], results_text, indices[0], distances[0], emb_sub)
    
    # 8) Formato final
    final_results = []
    for item in re_ranked:
        texto = item[0]
        idx_sub = item[1]
        dotp = item[3]
        row = df_sub_reset.iloc[idx_sub]
        circ = row.get("Circular", "N/A")
        fecha = row.get("Fecha", "N/A")
        final_results.append((texto, circ, fecha, dotp))

    return final_results

########################################################################
# Mensajes base con las instrucciones de Chain of Thought
########################################################################

@st.cache_data
def get_base_messages():
    """
    Aquí agregamos las instrucciones para usar Chain of Thought (CoT).
    """
    return [
         {"role": "system", "content": "Eres un asistente que da apoyo al área comercial. Responde claramente basándote en los contenidos proporcionados. Siempre indica de qué circular tomaste la información, como ejemplo 'información tomada de CIRCULAR NORMATIVA EXTERNA No. XXX DE XXX', si la respuesta es de varias circulares debes relacionarlas todas."}
        ,{"role": "system", "content": "Si te preguntan que cual es el contexto que tienes, debes responder que son todas las CNE emitidas en el 2024, que va desde la CIRCULAR NORMATIVA EXTERNA No. 001 DE 2024 - Ajuste al Programa Especial de Garantía Fusagasugá - EMP440 Créditos para la Gente hasta la CIRCULAR NORMATIVA EXTERNA N° 032 DE 2024 - Ajuste a la tarifa y cobertura a productos de garantía FNG... en total tienes 32 CNE"}
        ,{"role": "system", "content": "Primero, razona paso a paso (Chain of Thought) de manera interna, sin exponer tu razonamiento completo al usuario. Luego da al usuario solo la respuesta final, indicando siempre la circular o circulares utilizadas. "}
        ,{"role": "system", "content": "Solo puedes responder sobre temas asociados al Fondo Nacional de Garantías. Si te preguntan otros temas, responde: 'No puedo responder tu solicitud, mi conocimiento se basa únicamente en circulares del FNG emitidas en el 2024.'"}
#        ,{"role": "system", "content": "Si ves que tu respuesta se va a basar en una tabla siempre ten en cuenta toda la tabla para generar la respuesta, las tablas se encuentran en formato markdown, es necesario considerar toda la tabla correspondiente al producto que te esten preguntando"}
#        ,{"role": "system", "content": "Cuando tu respuesta dependa de una tabla, **revisa y considera todas las filas** que correspondan al rango de cobertura o tipo de producto que te estén preguntando (por ejemplo, Tipo I a Tipo XX). No omitas ni ignores filas. Si la pregunta involucra un rango de cobertura específico, asegúrate de identificar y utilizar **todas** las filas de la tabla que abarquen ese rango, y luego determina los valores mínimos y máximos según corresponda, es necesario que expongas toda la tabla para que internamente la proceses por medio Chain of Thought (Primero razona internamente bloque por bloque de la tabla)"}
        ,{"role": "system", "content": "Cuando tu respuesta dependa de una tabla sobre coberturas, comisiones o valores de cualquier producto (p. ej., EMP001, EMP101, etc.), debes:\n\n1) Identificar el producto y cobertura que se mencionan en la pregunta.\n2) Localizar la tabla correspondiente a ese producto.\n3) Considerar **todas** las filas que aplican a la cobertura indicada (p. ej. si la cobertura es >40% - ≤50%, revisa todas las filas de la tabla que cubran ese rango, sin omitir ninguna).\n4) Mostrar o usar la comisión mínima y la máxima encontradas, o los valores solicitados, asegurándote de incluir toda la tabla para tu análisis.\n5) Citar la(s) circular(es) o referencia(s) de donde obtuviste la información.\n\nNo omitas ni ignores filas, ni te limites a la primera coincidencia."}
    ]

########################################################################
# Función principal del chatbot
########################################################################

def run_chatbot():
    logo_path = "Imagen2.png"

    with st.container():
        col1, col2 = st.columns([1, 5])

        with col1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(logo_path, width=120)

        with col2:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h1 style='color:gray; font-size: 1.8em;'>🤖 Sub. Arquitectura de datos<br>Asistente CNE Garant-IA 💭</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

    if "messages" not in st.session_state:
        st.session_state.messages = get_base_messages()

    # Mostrar historial
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"**Tú:** {content}")
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**Respuesta Garant-IA:** {content}")

    query = st.chat_input("¿En qué puedo ayudarte?")
    if query:
        # Agregamos el mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**Tú:** {query}")
            
        typing_placeholder = st.empty()
        for _ in range(3):
            for dots in ["", ".", "..", "..."]:
                typing_placeholder.markdown(f"**Garant-IA está escribiendo{dots} 🤔**")
                time.sleep(0.2)

        # >>> Búsqueda semántica con FAISS <<<
        faiss_results = search_similar(query, k=10)

        # Armamos el contexto
        context = "\n".join([
            f"Circular: {c}, Fecha: {f}, Texto: {t}"
            for (t, c, f, score) in faiss_results
        ])

        # Insertamos el contexto como nuevo mensaje system
        st.session_state.messages.append({
            "role": "system",
            "content": (
                f"Documentos relevantes:\n{context}\n\n"
                "Primero razona internamente paso a paso (Chain of Thought) y luego responde con la conclusión final, citando la circular o circulares relevantes, no debes mostrar la cadena de pensamiento, solo el resutado final"
            )
        }
        )
        st.session_state.messages.append(
        {"role": "system", "content": "Cuando tu respuesta dependa de una tabla sobre coberturas, comisiones o valores de cualquier producto (p. ej., EMP001, EMP101, etc.), debes:\n\n1) Identificar el producto y cobertura que se mencionan en la pregunta.\n2) Localizar la tabla correspondiente a ese producto.\n3) Considerar **todas** las filas que aplican a la cobertura indicada (p. ej. si la cobertura es >40% - ≤50%, revisa todas las filas de la tabla que cubran ese rango, sin omitir ninguna).\n4) Mostrar o usar la comisión mínima y la máxima encontradas, o los valores solicitados, asegurándote de incluir toda la tabla para tu análisis.\n5) Citar la(s) circular(es) o referencia(s) de donde obtuviste la información.\n\nNo omitas ni ignores filas, ni te limites a la primera coincidencia."}
        )


        # Llamada a OpenAI Chat
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # Ajusta según tu caso
            messages=st.session_state.messages,
            temperature=0.5
        )
        content = response.choices[0].message.content

        # Guardamos la respuesta en el historial
        st.session_state.messages.append({"role": "assistant", "content": content})

        typing_placeholder.empty()
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            partial = ""
            for ch in content:
                partial += ch
                placeholder.markdown(f"**Respuesta Garant-IA:** {partial}")
                time.sleep(0.005)
            placeholder.markdown(f"**Respuesta Garant-IA:** {partial}")

if __name__ == "__main__":
    run_chatbot()
