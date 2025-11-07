from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import fitz  # PyMuPDF
import uuid
import json
import requests

app = FastAPI(title="Local RAG API")

# 초기화: 임베딩 모델 & Qdrant 클라이언트
model = SentenceTransformer("BAAI/bge-m3")
qdrant = QdrantClient(host="qdrant", port=6333)

# manuals 컬렉션이 없으면 생성
try:
    qdrant.get_collection("manuals")
except Exception:
    qdrant.recreate_collection(
        collection_name="manuals",
        vectors_config=VectorParams(size=1024, distance="Cosine")
    )

# PDF → 텍스트 추출 함수
def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text("text") + "\n"
    return text

# 텍스트 청크 분할 함수
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

# /ingest - PDF 업로드 → 벡터DB 저장
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, show_progress_bar=True)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": chunks[i]}
        )
        for i in range(len(chunks))
    ]

    qdrant.recreate_collection(
        collection_name="manuals",
        vectors_config=VectorParams(size=len(embeddings[0]), distance="Cosine")
    )

    qdrant.upload_points(collection_name="manuals", points=points)
    return {"status": "ok", "chunks_saved": len(chunks)}


# /ask - 질의 → Qdrant 검색 → Ollama 전달
@app.post("/ask")
async def ask(query: str = Form(...)):
    # 질의 임베딩
    q_emb = model.encode([query])[0]

    # Qdrant에서 관련 문서 검색
    results = qdrant.search(collection_name="manuals", query_vector=q_emb, limit=3)

    # 검색 결과 텍스트 병합
    context = "\n".join([r.payload["text"] for r in results])

    print("\n===== [검색된 컨텍스트] =====")
    print(context)
    print("============================\n")

    # LLM 프롬프트 구성
    prompt = f"""
	모든 답변은 반드시 한국어로 작성하라.
	너는 논문을 잘 이해하고 설명하는 전문가이다.
	아래 참고내용을 바탕으로 사용자의 질문에 명확하고 간결하게 답변하라.
	
	- 답변은 반드시 한국어로 자연스럽고 명확하게 작성할 것.
	- 논리적이고 간결하게 요약할 것.
	- 기술용어는 한국어로 번역하되, 원어(영문)도 괄호 안에 함께 표기할 것.
	[참고 내용]
    {context}

    [질문]
    {query}
    """

    # Ollama REST API 호출
    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "llama3:latest", "prompt": prompt},
        stream=True
    )

    if response.status_code != 200:
        return {"error": response.text}

	# Ollama 스트리밍 JSON 파싱
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

    # 개행문자 및 불필요한 이스케이프 제거
    output = output.replace("\\n", " ").replace("\n", " ").strip()

    if not output.strip():
        return {"error": "응답이 비어 있습니다. 모델 이름 또는 서버 상태를 확인하세요."}

    return JSONResponse(content={"answer": output})

# Qdrant 데이터 확인용 API
@app.get("/collections")
async def list_collections():
    return qdrant.get_collections()

@app.get("/collection/{name}/points")
async def show_points(name: str, limit: int = 5):
    result = qdrant.scroll(collection_name=name, limit=limit)
    return result
