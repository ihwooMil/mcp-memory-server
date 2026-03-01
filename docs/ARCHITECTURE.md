# Long-Term Memory Architecture

> long-term-memory 내부 구조와 설계 결정을 설명하는 기술 문서

## 시스템 개요

Long-Term Memory는 AI 어시스턴트에 **영구적이고 자기 관리되는 기억**을 부여하는 MCP 서버입니다.

기존 `.md` 파일 기반 기억의 두 가지 한계를 해결합니다:

1. **수동 검색 문제** — `.md`에 적어둬도 AI가 매번 확인하지 않음
2. **토큰 낭비 문제** — 전체 파일을 context에 넣으면 불필요한 내용까지 포함

## 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────┐
│                    MCP Client                         │
│           (Claude Desktop / Claude Code)              │
└──────────────────────┬───────────────────────────────┘
                       │ stdio (JSON-RPC)
┌──────────────────────▼───────────────────────────────┐
│               FastMCP Server (12 tools)               │
│               src/aimemory/mcp/server.py              │
├──────────────────────────────────────────────────────┤
│               MemoryBridge (orchestrator)              │
│               src/aimemory/mcp/bridge.py               │
├─────────┬───────────┬───────────┬────────────────────┤
│ Decision│ Retrieval │ Storage   │ Maintenance         │
│ Layer   │ Layer     │ Layer     │ Layer               │
├─────────┼───────────┼───────────┼────────────────────┤
│ Online  │ Graph     │ Graph     │ Sleep Cycle         │
│ Policy  │ Retriever │ Memory    │ Runner              │
│ + Agent │ (GraphRAG)│ Store     │                     │
│         │           │ (ChromaDB)│ ├─ Consolidation    │
│ ReRanker│ Knowledge │           │ ├─ Forgetting       │
│ (11d)   │ Graph     │           │ ├─ Resolution Regen │
│         │ (NetworkX)│           │ └─ Checkpoint       │
│ Feedback│           │           │                     │
│ Detector│ Context   │           │                     │
│         │ Composer  │           │                     │
└─────────┴───────────┴───────────┴────────────────────┘
```

## 6개 레이어 상세

---

### Layer 1: Decision — RL 메모리 정책

**파일**: `src/aimemory/online/policy.py`, `enhanced_policy.py`

매 대화 턴마다 **SAVE / SKIP / RETRIEVE** 중 하나를 결정합니다.

#### 3단계 결정 파이프라인

```
사용자 메시지 입력
    │
    ▼
Phase 0: RETRIEVE 판단 (rule-based)
    질문 패턴? discourse marker? → 저장된 메모리 있으면 RETRIEVE
    │
    ▼
Phase 1: 고/저확신 직접 결정 (rule-based)
    importance ≥ 0.7 → SAVE (개인정보, 강한 선호 등)
    importance ≤ 0.1 → SKIP (인사, 짧은 응답 등)
    │
    ▼
Phase 2: 중간 영역 → MLP Bandit 결정
    10d feature vector → Linear(10,64) → ReLU → Linear(64,3)
    epsilon-greedy (ε=0.1) action selection
```

#### Importance 가중합

| 패턴 | 가중치 | 예시 |
|------|--------|------|
| 개인정보 | +0.4 | "내 이름은...", "나는 서울에 살아" |
| 선호 | +0.35 | "나는 Python을 좋아해" |
| 기술 | +0.3 | "React 18에서 Suspense는..." |
| 감정 | +0.2 | "요즘 번아웃이 심해" |
| 키워드 밀도 | +0.15 | 명사 비율이 높은 문장 |

#### StateEncoder (10d feature vector)

```
[turn_position, memory_count, keyword_count, is_question,
 has_personal_info, has_preference, has_tech, has_emotion,
 recent_save_count, recent_retrieve_count]
```

#### Enhanced Policy (opt-in)

- 778d state: SentenceTransformer 768d + hand-crafted 10d
- 더 큰 MLP: 778d → 256 → 128 → 3 (dropout 0.1)
- Experience Replay Buffer (capacity=5000, batch=32)
- Progressive Autonomy: 긍정 피드백 누적 → RL 영역 점진적 확장

---

### Layer 2: Retrieval — GraphRAG 하이브리드 검색

**파일**: `src/aimemory/memory/graph_retriever.py`, `online/reranker.py`

#### 검색 파이프라인

```
사용자 메시지
    │
    ├─→ ChromaDB 벡터 검색 (cosine similarity)
    │   상위 20개 후보 (reranker_pool_size)
    │
    ├─→ KnowledgeGraph 탐색 (opt-in)
    │   엔티티 추출 → multi-hop traversal
    │
    ▼
Score Fusion: final = vector × 0.6 + graph × 0.4
    │
    ▼
RL Re-ranker (11d feature → 32h → 1 score)
    │
    ▼
Context Composer (token budget 내 top-K 선별)
```

#### Re-ranker Features (11d)

| # | Feature | 설명 |
|---|---------|------|
| 0 | chroma_similarity | 원본 벡터 유사도 |
| 1 | keyword_overlap | 쿼리-메모리 키워드 Jaccard |
| 2 | category_match | 추론된 카테고리 일치 |
| 3 | recency | 카테고리별 시간 감쇠 |
| 4 | access_frequency | log1p(access_count) |
| 5 | content_length_ratio | 정규화된 길이 비율 |
| 6 | has_related | 그래프 연결 여부 |
| 7 | resolution_available | 가용 해상도 수 |
| 8 | graph_connection_count | KG 엔티티 연결 수 |
| 9 | graph_hop_distance | KG 홉 거리 |
| 10 | has_negative_relation | 부정 관계 여부 |

#### Context Composer — 다해상도 선택

토큰 budget(기본 1024) 내에서 MMR(Maximal Marginal Relevance) 알고리즘으로 다양성과 관련성을 균형 있게 선택합니다.

```
메모리 A: Level 0 (전문 60토큰) → budget 초과 시 Level 1 (요약 25토큰) → Level 2 (트리플 10토큰)
메모리 B: Level 0 (전문 40토큰) → budget 내이면 그대로 사용
```

3가지 해상도:
- **Level 0**: 원문 그대로
- **Level 1**: 키워드 포함 1-2문장 요약 (max 100자)
- **Level 2**: entity triple (subject, predicate, object)

---

### Layer 3: Storage — GraphMemoryStore

**파일**: `src/aimemory/memory/graph_store.py`, `knowledge_graph.py`

#### MemoryNode 구조

```python
MemoryNode(
    memory_id="a1b2c3d4e5f6",     # 12-char hex
    content="사용자는 Python을 좋아함",
    keywords=["Python", "선호"],
    category="preference",          # fact/preference/experience/emotion/technical/core_principle
    related_ids=["x1y2z3..."],      # 양방향 그래프 엣지
    created_at="2026-02-28T...",
    access_count=5,                 # 검색 시 자동 증가
    level1_text="Python 선호",      # Level 1 요약
    level2_text="사용자,좋아함,Python",  # Level 2 트리플
    immutable=False,                # True이면 수정/삭제 불가
    pinned=False,                   # True이면 망각 보호
    active=True,                    # False이면 검색 제외
)
```

#### ChromaDB 설정

- Embedding model: `intfloat/multilingual-e5-small` (384d, 한/영 지원)
- Distance metric: cosine (HNSW)
- 영속 저장: SQLite 기반 (서버 재시작 후에도 유지)

#### KnowledgeGraph (NetworkX DiGraph)

Level 2 텍스트(`subject,predicate,object`)를 자동 파싱하여 엔티티-관계 그래프를 구축합니다.

```
사용자 ──좋아함──→ Python
사용자 ──싫어함──→ Java
Python ──관련──→ FastAPI
```

- multi-hop 탐색 (깊이 1~3)
- 부정 관계 감지 ("싫어함", "dislike" 등)
- 메모리 저장/삭제 시 자동 동기화

---

### Layer 4: Maintenance — 수면 사이클

**파일**: `src/aimemory/memory/sleep_cycle.py`, `forgetting.py`, `consolidation.py`

주기적으로 실행되는 4단계 유지보수:

```
Sleep Cycle
    │
    ├─ 1. Consolidation (통합)
    │     유사도 ≥ 0.92인 메모리 쌍 탐지 → 병합
    │     access_count 높은 쪽이 생존
    │
    ├─ 2. Resolution Regeneration (해상도 재생성)
    │     level1_text / level2_text 누락된 메모리 재생성
    │
    ├─ 3. Forgetting (망각)
    │     importance = (1 + access_count) × e^(-λ×days) + related_boost × edges
    │
    │     importance < 0.3 → Level 2로 압축
    │     importance < 0.1 → 비활성화 (검색 제외)
    │     비활성 30일 경과 → 영구 삭제 + 감사 기록
    │
    │     ※ pinned / immutable 메모리는 보호됨
    │
    └─ 4. Checkpoint (체크포인트)
          RL 모델 파라미터 저장
```

---

### Layer 5: Reward — 보상 신호

**파일**: `src/aimemory/reward/`, `online/reranker.py`

RL 정책을 실시간으로 개선하기 위한 보상 수집:

#### 명시적 피드백 (FeedbackDetector)

한국어/영어 패턴으로 사용자 피드백 감지:
- 긍정: "맞아", "좋아", "exactly", "perfect" → +1.0
- 부정: "아니야", "틀렸어", "wrong", "no" → -1.0

#### 암묵적 보상 (ImplicitRewardDetector)

대화 흐름 패턴으로 보상 추론:
- 대화 지속 (2+ 턴 이어감) → +0.3
- 화제 확장 (기억 키워드 재등장) → +0.2
- 짧은 무관심 응답 → -0.1

---

### Layer 6: P2P Federated Learning (opt-in)

**파일**: `src/aimemory/online/gossip.py`, `transport.py`, `rule_verifier.py`

여러 사용자의 RL 정책을 P2P로 공유하여 학습 속도 향상:

```
Node A ←──gossip──→ Node B ←──gossip──→ Node C
  │                    │                    │
  └── 각자의 OnlinePolicy 파라미터 교환 ──┘
```

#### 보안

- **L2 Norm Clipping**: 파라미터 업데이트 크기 제한
- **Differential Privacy**: Gaussian noise 추가 (선택적)
- **Krum Aggregation**: Byzantine-tolerant 집계 (악의적 노드 필터링)
- **Rule Hash Verification**: SHA-256으로 보안 규칙 변조 탐지

---

## MCP 도구 (12개)

| 도구 | 레이어 | 설명 |
|------|--------|------|
| `auto_search` | Retrieval | 매 턴 관련 기억 자동 검색 + 다해상도 컨텍스트 조합 |
| `memory_save` | Storage | 새 기억 저장 (키워드, 카테고리, 관계 지정) |
| `memory_search` | Retrieval | 시맨틱 유사도 검색 |
| `memory_update` | Storage | 기존 기억 내용/키워드 수정 |
| `memory_delete` | Storage | 기억 삭제 (immutable 존중) |
| `memory_get_related` | Retrieval | BFS 그래프 탐색 (depth 1~3) |
| `memory_pin` | Storage | 망각 보호 설정 |
| `memory_unpin` | Storage | 망각 보호 해제 |
| `memory_stats` | Storage | 총 개수 + 카테고리별 분포 |
| `sleep_cycle_run` | Maintenance | 수면 사이클 실행 (통합+망각+체크포인트) |
| `policy_status` | Decision | RL 정책 상태 (epsilon, action 분포, 업데이트 수) |
| `policy_decide` | Decision | RL 정책 결정 트레이스 (SAVE/SKIP/RETRIEVE + 이유) |

---

## 설정

### 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `AIMEMORY_DB_PATH` | `./memory_db` | ChromaDB 저장 경로 |
| `AIMEMORY_LANGUAGE` | `ko` | 패턴 매칭 언어 (`ko`/`en`) |
| `AIMEMORY_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | 임베딩 모델 |
| `AIMEMORY_LOG_LEVEL` | `INFO` | 로깅 레벨 |
| `AIMEMORY_ENHANCED_POLICY` | `0` | Enhanced RL 정책 활성화 |
| `AIMEMORY_GRAPH_RAG` | `0` | GraphRAG 하이브리드 검색 활성화 |

### Config 클래스 체계

```
AppConfig
├── OnlinePolicyConfig   — RL 정책 (thresholds, epsilon, lr)
├── ReRankerConfig       — Re-ranker (feature_dim, latency budget)
├── MCPServerConfig      — MCP 서버 (token_budget, top_k)
├── ForgettingConfig     — 망각 파이프라인 (thresholds, decay_lambda)
├── ComposerConfig       — Context Composer (budget, level별 평균 토큰)
├── SleepCycleConfig     — 수면 사이클 (간격, 리포트 저장)
├── GossipConfig         — P2P 학습 (interval, DP 설정)
├── SecurityConfig       — 불변 규칙 (비밀번호/API키 차단)
├── RewardConfig         — 보상 가중치
├── SelfPlayConfig       — Self-play 시뮬레이션
├── OllamaConfig         — 로컬 LLM (모델명, max_tokens)
├── DatasetConfig        — 데이터셋 분할 비율
└── DataPaths            — 파일 경로
```

---

## 기술 스택

| 영역 | 기술 | 용도 |
|------|------|------|
| MCP 프로토콜 | FastMCP | MCP 서버 구현 |
| 벡터 DB | ChromaDB | 시맨틱 검색 + 영속 저장 |
| 임베딩 | sentence-transformers | 다국어 텍스트 임베딩 |
| 지식 그래프 | NetworkX | 엔티티-관계 그래프 |
| 데이터 모델 | Pydantic v2 | 스키마 검증 + 직렬화 |
| RL | PyTorch (via MLP) | 순수 Python MLP (경량) |
| 테스트 | pytest | 590개 테스트 |
| 린트 | ruff | 포맷팅 + 린팅 |
| CI | GitHub Actions | Python 3.11/3.12/3.13 매트릭스 |
| 패키지 | hatchling + uv | 빌드 + 의존성 관리 |

---

## 설계 원칙

1. **"저장 품질 > 검색 알고리즘"** — 무엇을 저장할지가 가장 중요
2. **Additive 확장** — 기존 클래스 수정 최소화, opt-in config flags
3. **Graceful Degradation** — Enhanced/GraphRAG 비활성 시 기본 모드로 동작
4. **로컬 우선** — 외부 API 불필요, 모든 데이터 로컬 저장
5. **다국어 설계** — i18n 모듈로 언어별 패턴 분리
