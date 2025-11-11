from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import fitz
import uuid
import json
import requests
import time
import os
import csv
from datetime import datetime
import numpy as np

app = FastAPI(title="Local RAG API with Metrics & Logging")

#로컬용 설정
QDRANT_HOST = "localhost"   
QDRANT_PORT = 6333
OLLAMA_API = "http://localhost:11434/api/generate"  

# 임베딩 모델, Qdrant 초기화
#model = SentenceTransformer("BAAI/bge-m3")
model  = SentenceTransformer("nlpai-lab/KURE-v1")
embedding_model_name = "nlpai-lab/KURE-v1"
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

#로그 저장 디렉토리 설정
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_logs.csv")
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "query", "elapsed_time_sec",
            "avg_similarity", "embedding_model", "llm_model", "top_contexts", "answer_summary"
        ])

# Qdrant 초기 설정
try:
    qdrant.get_collection("manuals")
except Exception:
    qdrant.recreate_collection(
        collection_name="manuals",
        vectors_config=VectorParams(size=1024, distance="Cosine")
    )

# PDF 텍스트 추출
def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text("text") + "\n"
    return text

# 텍스트 분할
def chunk_text(text, max_len=500):
    sentences = text.split("\n")
    chunks, current_chunk = [], ""
    for sent in sentences:
        if len(current_chunk) + len(sent) > max_len:
            chunks.append(current_chunk.strip())
            current_chunk = sent
        else:
            current_chunk += " " + sent
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

#pdf 벡터화, db 저장
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, show_progress_bar=True)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "text": chunks[i],
                "source": file.filename
            }
        )
        for i in range(len(chunks))
    ]

    qdrant.upload_points(collection_name="manuals", points=points)
    return {"status": "ok", "file": file.filename, "chunks_saved": len(chunks)}


@app.post("/ask")
async def ask(query: str = Form(...)):
    start_time = time.time()
    model_name = "llama3:latest"
    
    #쿼리 임베딩
    q_emb = model.encode([query])[0]
    
    #qdrant에서 유사 문서 검색
    results = qdrant.search(collection_name="manuals", query_vector=q_emb, limit=3)
    
    #평균 유사도 계산
    avg_similarity = np.mean([r.score for r in results]) if results else 0.0
    #avg_similarity = round(sum(similarities) / len(similarities), 4) if similarities else 0.0


    #검색결과 병합
    context = ""
    similarities = []
    top_contexts = []

    for r in results:
        src = r.payload.get("source", "unknown")
        top_contexts.append(src)
        context += f"\n[출처: {src}]\n{r.payload['text']}\n"
        sim = util.cos_sim(q_emb, model.encode(r.payload["text"]))[0][0].item()
        similarities.append(sim)


    prompt = f"""
    모든 답변은 반드시 한국어로 작성하라.
    너는 논문을 잘 이해하고 설명하는 전문가이다.
    아래 참고내용을 바탕으로 사용자의 질문에 명확하고 간결하게 답변하라.

    [참고 내용]
    {context}

    [질문]
    {query}
    """

    #Ollama 로컬 REST API 호출
    response = requests.post(
        OLLAMA_API,
        json={"model": model_name, "prompt": prompt},
        stream=True
    )

    if response.status_code != 200:
        return {"error": response.text}

    output = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                for key in ["response", "message", "content"]:
                    if key in data:
                        output += data[key]
            except Exception:
                continue

    output = output.replace("\\n", "\n").strip()
    elapsed_time = round(time.time() - start_time, 2)

    if not output.strip():
        return {"error": "응답이 비어 있습니다."}

    # 로그 저장
    output_clean = output.replace("\n"," ")

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            elapsed_time,
            avg_similarity,
            embedding_model_name,
            model_name,
            "; ".join(top_contexts),
            output_clean # 답변 전체 기록
        ])

    return JSONResponse(
        content={
            "answer": output,
            "elapsed_time_sec": elapsed_time,
            "avg_similarity": avg_similarity,
            "embedding_model" : embedding_model_name,
            "logged": True
        }
    )


#qdrant 상태 확인
@app.get("/collections")
async def list_collections():
    return qdrant.get_collections()


@app.get("/collection/{name}/points")
async def show_points(name: str, limit: int = 5):
    result = qdrant.scroll(collection_name=name, limit=limit)
    return result
