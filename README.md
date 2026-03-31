# 🤖 Garant-IA — Asistente Inteligente de Circulares Normativas (FNG)

> Chatbot conversacional basado en RAG (Retrieval-Augmented Generation) para consultar y analizar Circulares Normativas Externas (CNE) del **Fondo Nacional de Garantías**, construido con FAISS, OpenAI Embeddings y Streamlit.

---

## 📌 ¿Qué hace este proyecto?

**Garant-IA** permite a equipos comerciales y operativos consultar en lenguaje natural el contenido de las 32 Circulares Normativas Externas (CNE) emitidas por el FNG en 2024, sin necesidad de buscar manualmente en documentos PDF o Excel.

El sistema:
- Recibe una pregunta en lenguaje natural (ej: *"¿Cuál es la cobertura del EMP440 para créditos menores a 10 millones?"*)
- Busca semánticamente los fragmentos más relevantes de las circulares usando **FAISS + embeddings de OpenAI**
- Genera una respuesta precisa citando siempre la circular fuente
- Aplica **Chain of Thought (CoT)** internamente para razonar sobre tablas de coberturas y comisiones

---

## 🧠 Arquitectura del sistema

```
Usuario (pregunta natural)
        │
        ▼
[Detección de producto EMP]
        │
        ▼
[Filtrado semántico por producto]
        │
        ▼
[FAISS Index — búsqueda por similitud coseno]
        │
        ▼
[Reranking local por dot product]
        │
        ▼
[Contexto enriquecido → GPT-4o-mini + Chain of Thought]
        │
        ▼
Respuesta con cita de circular(es)
```

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| Interfaz web | Streamlit |
| Búsqueda semántica | FAISS (IndexFlatIP) |
| Embeddings | OpenAI `text-embedding-3-large` |
| Generación de respuestas | OpenAI `GPT-4o-mini` |
| Razonamiento | Chain of Thought (CoT) via prompting |
| Base de conocimiento | Excel chunkeado + embeddings precalculados |
| Preprocesamiento | NLTK, Unidecode, Pandas |

---

## 📁 Estructura del proyecto

```
Proyecto-Garant-IA/
│
├── Streamlit_cargar.py     # Aplicación principal
├── Chunks3.xlsx            # Base de conocimiento (circulares chunkeadas)
├── embeddings.npy          # Embeddings precalculados (text-embedding-3-large)
├── Imagen2.png             # Logo institucional
├── requirements.txt        # Dependencias del proyecto
└── .env                    # Variables de entorno (API Key - NO subir a GitHub)
```

---

## ⚙️ Instalación y uso local

### 1. Clona el repositorio
```bash
git clone https://github.com/LuisCa-Cyber/Proyecto-Garant-IA.git
cd Proyecto-Garant-IA
```

### 2. Instala las dependencias
```bash
pip install -r requirements.txt
```

### 3. Configura tu API Key de OpenAI
Crea un archivo `.env` en la raíz del proyecto:
```
OPENAI_API_KEY=tu_api_key_aqui
```

### 4. Ejecuta la aplicación
```bash
streamlit run Streamlit_cargar.py
```

---

## 🔍 Características técnicas destacadas

- **Filtrado por producto:** detecta automáticamente códigos `EMPXXX` en la query y filtra el índice FAISS solo sobre los chunks relevantes, mejorando precisión y velocidad.
- **Reranking local:** aplica un segundo ordenamiento por dot product sobre los resultados de FAISS para maximizar relevancia.
- **Prompting estructurado con CoT:** el sistema razona internamente sobre tablas de coberturas antes de responder, evitando omisiones en rangos de comisiones.
- **Caché inteligente:** usa `@st.cache_data` para evitar recargar embeddings y datos en cada interacción.

---

## 📊 Cobertura de conocimiento

El asistente responde únicamente sobre las **32 CNE del FNG emitidas en 2024**, desde:
- `CNE No. 001/2024` — Ajuste al Programa Especial de Garantía Fusagasugá (EMP440)
- hasta
- `CNE No. 032/2024` — Ajuste a tarifa y cobertura de productos FNG

---

## 🚀 Posibles extensiones

- [ ] Agregar soporte para circulares de años anteriores (2022, 2023)
- [ ] Implementar reranking con un modelo cross-encoder
- [ ] Agregar memoria conversacional persistente
- [ ] Despliegue en Streamlit Cloud o AWS

---

## 👤 Autor

**Luis Fernández**  
Magíster en Inteligencia Artificial — Pontificia Universidad Javeriana  
[GitHub](https://github.com/LuisCa-Cyber) · [LinkedIn](https://linkedin.com/in/tu-perfil)

---

## 📄 Licencia

Este proyecto fue desarrollado como solución interna para automatizar la consulta de normativa del FNG. El código es de uso educativo y de portafolio.
