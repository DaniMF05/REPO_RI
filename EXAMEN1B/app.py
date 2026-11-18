import pickle
from typing import Literal, List
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from preprocesamiento import preprocess

app = FastAPI(title="Sistema RI - SciFact (ICCD753)")

# Cargar artefactos en memoria al arrancar el servidor
with open("ir_artifacts.pkl", "rb") as f:
    artefactos = pickle.load(f)

docs_ids = artefactos["docs_ids"]
metadata = artefactos["metadata"]
vectorizador = artefactos["vectorizador"]
matriz_tfidf = artefactos["matriz_tfidf"]
bm25_data = artefactos["bm25"]


# Esquemas de validación de entrada y salida
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: Literal["bm25", "tfidf"]
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResultItem(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    model: str
    results: List[SearchResultItem]


def buscar_tfidf(query: str, top_k: int) -> List[tuple]:
    vec_q = vectorizador.transform([query])
    scores = cosine_similarity(vec_q, matriz_tfidf).flatten()
    indices_validos = np.where(scores > 0.0)[0]
    resultados = [(docs_ids[i], float(scores[i])) for i in indices_validos]
    resultados.sort(key=lambda x: (-x[1], x[0]))
    return resultados[:top_k]


def buscar_bm25(query: str, top_k: int) -> List[tuple]:
    tokens = preprocess(query)
    inv_index = bm25_data["inverted_index"]
    idf = bm25_data["idf"]
    doc_len = bm25_data["doc_len"]
    avgdl = bm25_data["avgdl"]
    k1 = bm25_data["k1"]
    b = bm25_data["b"]

    scores_acumulados = {}
    for t in tokens:
        if t in inv_index:
            idf_t = idf[t]
            postings = inv_index[t]
            for doc_id, f in postings.items():
                dl = doc_len[doc_id]
                numerador = f * (k1 + 1.0)
                denominador = f + k1 * (1.0 - b + b * (dl / avgdl))
                score_termino = idf_t * (numerador / denominador)
                scores_acumulados[doc_id] = scores_acumulados.get(doc_id, 0.0) + score_termino

    resultados = [(doc_id, float(sc)) for doc_id, sc in scores_acumulados.items() if sc > 0.0]
    resultados.sort(key=lambda x: (-x[1], x[0]))
    return resultados[:top_k]


@app.get("/health")
def health_check():
    return {"status": "ok", "corpus": "SciFact", "documents_indexed": len(docs_ids)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía.")

    if req.model == "tfidf":
        raw_results = buscar_tfidf(req.query, req.top_k)
    else:
        raw_results = buscar_bm25(req.query, req.top_k)

    formatted_results = []
    for idx, (doc_id, score) in enumerate(raw_results, start=1):
        doc_meta = metadata.get(doc_id, {"title": "", "snippet": ""})
        formatted_results.append(
            SearchResultItem(
                rank=idx,
                document_id=doc_id,
                score=round(score, 4),
                title=doc_meta["title"],
                snippet=doc_meta["snippet"]
            )
        )

    return SearchResponse(model=req.model, results=formatted_results)