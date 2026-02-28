# AIMemory 프로젝트 정리

> 작성일: 2026-02-28 | v0.1.0

## 1. 프로젝트 개요

**AI Memory System** — AI 챗봇이 사용자의 정보를 자동으로 기억하고 활용하는 시스템.

핵심 문제 해결:
- **수동 검색 문제**: 사용자가 명시적으로 요청해야만 메모리를 활용 → 매 턴 자동 검색
- **토큰 낭비 문제**: 전체 기억을 매번 context에 삽입 → top-K 선별 + 다해상도 압축

## 2. 시스템 아키텍처

```
사용자 메시지
    │
    ▼
┌─────────────────────────┐
│  State Encoder (394d)    │  텍스트 임베딩(384d) + 수작업 특징(10d)
│  → Action Selection      │  SAVE / SKIP / RETRIEVE (Contextual Bandit)
└─────────────┬───────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 SAVE      SKIP    RETRIEVE
    │                   │
    ▼                   ▼
┌────────┐      ┌──────────────┐
│GraphRAG│      │Context       │
│Store   │◄────►│Composer      │
│(Chroma)│      │(Token Budget)│
└────────┘      └──────────────┘
    │
    ▼ (주기적)
┌─────────────┐
│Sleep Cycle   │  망각 · 통합 · 재인코딩
└─────────────┘
```

### 6 레이어

| 레이어 | 모듈 | 역할 |
|--------|------|------|
| 1. Policy | `online/policy.py`, `online/enhanced_policy.py` | Feature → Action (SAVE/SKIP/RETRIEVE) |
| 2. Rule Filter | `online/rule_verifier.py` | 불변 규칙 (보안/프라이버시) |
| 3. Storage | `memory/graph_store.py`, `memory/knowledge_graph.py` | ChromaDB + Knowledge Graph |
| 4. Retrieval | `memory/graph_retriever.py`, `online/reranker.py` | Vector search + RL Re-ranking |
| 5. Composer | `memory/composer.py`, `memory/resolution.py` | 토큰 budget 내 최적 조합 + 다해상도 압축 |
| 6. Sleep Cycle | `memory/sleep_cycle.py`, `memory/forgetting.py`, `memory/consolidation.py` | 주기적 기억 정리 |

## 3. 모듈 구조

```
src/aimemory/               # 9,709 lines
├── __init__.py
├── config.py               # 전체 설정 (Pydantic)
├── schemas.py              # 데이터 모델 (Turn, Episode, MemoryActionType)
│
├── i18n/                   # 다국어 지원
│   ├── __init__.py         # LanguagePatterns 레지스트리
│   ├── ko.py               # 한국어 패턴 (정규식 40+개)
│   └── en.py               # 영어 패턴
│
├── mcp/                    # MCP 서버 (12 tools)
│   ├── __init__.py
│   ├── __main__.py         # python -m aimemory.mcp
│   ├── server.py           # FastMCP 서버 정의
│   └── bridge.py           # Policy ↔ MCP 연결
│
├── memory/                 # 기억 저장/검색/관리
│   ├── graph_store.py      # ChromaDB + sentence-transformers
│   ├── knowledge_graph.py  # 트리플 기반 지식 그래프
│   ├── graph_retriever.py  # 하이브리드 검색 (Vector + Graph)
│   ├── composer.py         # Context 조합 (토큰 budget)
│   ├── resolution.py       # 다해상도 텍스트 (L0/L1/L2)
│   ├── forgetting.py       # 망각 파이프라인
│   ├── consolidation.py    # 기억 통합
│   └── sleep_cycle.py      # 수면 사이클 오케스트레이션
│
├── online/                 # RL 온라인 학습
│   ├── policy.py           # Contextual Bandit (10d → 3 actions)
│   ├── enhanced_policy.py  # Enhanced MLP (394d → 3 actions)
│   ├── enhanced_encoder.py # 384d embedding + 10d features
│   ├── replay_buffer.py    # Experience replay
│   ├── autonomy.py         # Progressive autonomy (신뢰도 기반)
│   ├── reranker.py         # RL Re-ranker (검색 결과 재정렬)
│   ├── ab_comparator.py    # A/B 비교 테스트
│   ├── gossip.py           # P2P Federated Learning (Gossip)
│   ├── rule_verifier.py    # 불변 규칙 필터
│   └── transport.py        # P2P 통신 계층
│
├── reward/                 # 보상 계산
│   ├── calculator.py       # 종합 보상 계산기
│   ├── feedback_detector.py # 사용자 피드백 감지 (긍정/부정/교정)
│   ├── implicit_detector.py # 암묵적 보상 감지
│   ├── korean_patterns.py  # (legacy) 한국어 패턴
│   └── signals.py          # 보상 신호 정의
│
├── dataset/                # 학습 데이터 파이프라인
│   ├── builder.py          # Episode → 학습 데이터 변환
│   ├── splitter.py         # Train/Val/Test 분리
│   └── stats.py            # 데이터셋 통계
│
├── extractor/              # Offline RL Feature Extractor
│   ├── encoder.py          # Feature 인코더
│   ├── model.py            # DualHeadDQN
│   ├── dataset.py          # DQN 학습 데이터
│   ├── policy.py           # 추출 정책
│   └── trainer.py          # 학습 루프
│
└── selfplay/               # Self-play 데이터 생성
    ├── engine.py            # Self-play 엔진
    ├── llm_client.py        # Ollama 클라이언트
    ├── memory_agent.py      # 메모리 에이전트
    └── scenarios.py         # 시나리오 관리
```

## 4. 테스트 현황

```
tests/                      # 8,491 lines, 619 tests
├── test_i18n.py            # 다국어 패턴 (ko/en)
├── test_mcp_server.py      # MCP 서버
├── test_mcp_bridge.py      # MCP 브릿지
├── test_mcp_e2e.py         # MCP E2E
├── test_graph_store.py     # ChromaDB 저장소
├── test_graph_retriever.py # 하이브리드 검색
├── test_knowledge_graph.py # 지식 그래프
├── test_online_policy.py   # Contextual Bandit
├── test_enhanced_policy.py # Enhanced Policy
├── test_enhanced_encoder.py # Enhanced Encoder
├── test_replay_buffer.py   # Replay Buffer
├── test_reranker.py        # Re-ranker
├── test_ab_comparator.py   # A/B 비교
├── test_gossip.py          # Gossip Protocol
├── test_transport.py       # P2P Transport
├── test_autonomy.py        # Progressive Autonomy
├── test_rule_verifier.py   # Rule Verifier
├── test_reward_calculator.py # 보상 계산
├── test_feedback_detector.py # 피드백 감지
├── test_implicit_detector.py # 암묵적 보상
├── test_resolution.py      # 다해상도 텍스트
├── test_composer.py        # Context Composer
├── test_forgetting.py      # 망각 파이프라인
├── test_sleep_cycle.py     # 수면 사이클
├── test_extractor.py       # Feature Extractor
├── test_e2e_enhanced.py    # Enhanced E2E
├── test_dataset_builder.py # 데이터셋 빌더
├── test_schemas.py         # 스키마
└── test_selfplay_engine.py # Self-play (ollama 필요, skip)
```

**결과**: 619 passed, 1 skipped (14.6s)

## 5. 핵심 기술

### 5.1 Contextual Bandit Policy
- **State**: 394d feature vector (384d multilingual embedding + 10d hand-crafted)
- **Actions**: SAVE(0), SKIP(1), RETRIEVE(2)
- **Reward**: 사용자 피드백 기반 (긍정 +1.0, 부정 -1.0, 암묵적 보상 포함)
- **학습**: Online MLP + Experience Replay + Epsilon-greedy

### 5.2 GraphRAG Storage
- **Vector Store**: ChromaDB + `intfloat/multilingual-e5-small` (384d)
- **Knowledge Graph**: 트리플 기반 (`subject → predicate → object`)
- **Hybrid Retrieval**: Vector similarity + Graph traversal 결합

### 5.3 다해상도 텍스트
- **L0**: 원본 텍스트
- **L1**: 핵심 정보 추출 (주어+술어)
- **L2**: 키워드 압축
- Context Composer가 토큰 budget에 맞게 적절한 레벨 선택

### 5.4 수면 사이클
- 주기적으로 실행: 망각 → 통합 → 재인코딩
- **망각**: 접근 빈도 낮은 기억 삭제 (pinned/immutable 보호)
- **통합**: 유사 기억 병합
- **재인코딩**: 임베딩 재생성

### 5.5 다국어 지원 (i18n)
- `LanguagePatterns` 데이터 클래스로 언어별 패턴 분리
- 현재 지원: 한국어 (`ko`), 영어 (`en`)
- `register()` 함수로 새 언어 추가 가능

## 6. MCP 서버 도구 (12개)

| 도구 | 설명 |
|------|------|
| `auto_search` | 매 턴 자동 기억 검색 |
| `memory_save` | 기억 저장 |
| `memory_search` | 명시적 기억 검색 |
| `memory_update` | 기억 수정 |
| `memory_delete` | 기억 삭제 |
| `memory_stats` | 기억 통계 |
| `memory_list` | 기억 목록 |
| `memory_get` | 기억 상세 조회 |
| `graph_query` | 지식 그래프 쿼리 |
| `graph_add_triple` | 트리플 추가 |
| `sleep_cycle` | 수면 사이클 수동 실행 |
| `policy_stats` | 정책 통계 |

## 7. 의존성

### Core
```
pydantic>=2.0
chromadb>=0.5.0
sentence-transformers>=3.0
mcp[cli]>=1.2.0
```

### Optional
```
[ko]   mecab-python3, mecab-ko-dic     # 한국어 형태소 분석
[dev]  pytest, ruff                      # 개발
[train] ollama, pandas, pyarrow, tqdm, matplotlib  # 학습
```

## 8. 설치 및 실행

```bash
# 기본 설치
uv sync

# 한국어 지원 포함
uv sync --extra ko

# 개발 환경
uv sync --extra dev --extra ko

# 테스트
uv run pytest tests/ -q

# MCP 서버 실행
uv run python -m aimemory.mcp

# OpenClaw 연동
bash scripts/install_openclaw.sh
```

## 9. 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `AIMEMORY_DB_PATH` | `./memory_db` | ChromaDB 저장 경로 |
| `AIMEMORY_LANGUAGE` | `ko` | 언어 (`ko`, `en`) |
| `AIMEMORY_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | 임베딩 모델 |
| `AIMEMORY_LOG_LEVEL` | `WARNING` | 로그 레벨 |
| `AIMEMORY_TOKEN_BUDGET` | `500` | 검색 토큰 budget |
| `AIMEMORY_EPSILON` | `0.15` | Epsilon-greedy 탐색률 |

## 10. 구현 이력

| Step | 내용 | 상태 |
|------|------|------|
| 1 | 학습 데이터 확보 (Self-play + 공개 데이터) | ✅ |
| 2 | Rule-Based Baseline + Online MLP Bandit | ✅ |
| 3 | 망각 파이프라인 + 다해상도 저장 | ✅ |
| 4 | 수면 사이클 + 기억 통합 | ✅ |
| 5 | RL Re-ranker | ✅ |
| 6 | P2P Federated Learning (Gossip) | ✅ |
| 7 | MCP 서버 패키징 | ✅ |
| 8 | GraphRAG Integration + Enhanced Policy | ✅ |
| 9 | 다국어(i18n) + 설치 파이프라인 + 배포 준비 | ✅ |

## 11. 배포 방법

```bash
# 1. 빌드
uv build
# → dist/aimemory-0.1.0.tar.gz
# → dist/aimemory-0.1.0-py3-none-any.whl

# 2. TestPyPI에 먼저 테스트
uv publish --publish-url https://test.pypi.org/legacy/

# 3. PyPI 배포
uv publish
# PyPI API token 필요: https://pypi.org/manage/account/token/

# 4. 설치 확인
pip install aimemory
```

## 12. 코드 규모

| 항목 | 수치 |
|------|------|
| 소스 코드 | 51개 파일, 9,709 줄 |
| 테스트 코드 | 31개 파일, 8,491 줄 |
| 테스트 수 | 619 passed, 1 skipped |
| 스크립트 | 14개 |
| MCP 도구 | 12개 |
| 지원 언어 | 2 (ko, en) |
