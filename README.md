# 4inlab RAG-Server

## 개요
이 프로젝트는 로컬 환경에서 RAG(Retrieval-Augmented Generation) 기반의 문서 검색 및 요약 기능을 제공합니다.
PDF 메뉴얼이나 논문을 업로드하면, Qdrant 벡터 DB에 인베딩되어 LLM(Ollama)으로 질의응답할 수 있습니다.

## 주요기능
- /ingest : PDF -> Text -> 벡터 DB 업로드
- /ask : 사용자 질문 -> Qrant 검색 -> LLM 응답
- /collections : Qrant 컬렉션 조회
- /collection/{name}/points : 포인트 조회

### Docker 환경
 docker compose up --build

### 버전
- python : 3.10.13

  
