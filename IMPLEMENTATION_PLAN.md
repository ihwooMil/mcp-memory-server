# AI Memory System - 구현 계획서

> **기반 문서**: ai_memory_system_design_v4.docx
> **생성일**: 2026-02-25
> **목표**: Rule-Based + Online Bandit Memory Policy + Intelligent Memory MCP 시스템 구현

---

## 프로젝트 개요

현재 AI 메모리 도구의 두 가지 핵심 한계를 해결하는 시스템:
1. **수동 검색 문제** - 사용자가 명시적으로 요청해야만 메모리를 활용하는 문제 → 매 턴 자동 검색으로 해결
2. **토큰 낭비 문제** - 전체 기억을 매번 context에 삽입하는 문제 → top-K 선별 삽입으로 해결

### 시스템 아키텍처 (6 레이어)

| 레이어 | 역할 |
|--------|------|
| Rule-Based + Online Bandit | 대화에서 feature vector 생성 + 저장할 fact 결정 |
| Immutable Rule Filter | 불변 규칙 적용 (보안/프라이버시) |
| 저장소 | feature vector + fact/텍스트 저장 |
| 검색 | feature 유사도 매칭으로 관련 fact 조회 |
| Context Composer | 토큰 budget 내 최적 조합 |
| 수면 사이클 | 주기적 기억 정리/재인코딩/통합/망각 |

---

## Step 1: 학습 데이터 확보 ✅ 완료

> **목표**: RL Feature Extractor 학습에 필요한 데이터셋 확보
> **성공률**: ~100%
> **성공 기준**: reward 분포가 의미있게 분리됨 (positive/negative 골고루)
> **완료일**: 2026-02-26

### 1.1 합성 데이터 생성 파이프라인 ✅

- [x] 로컬 LLM 환경 세팅 (Qwen2.5-7B)
- [x] Self-play 시뮬레이션 프레임워크 설계 (`src/aimemory/selfplay/`)
  - [x] 대화 시나리오 템플릿 정의 (일상대화, 기술질문, 프로젝트 관련 등)
  - [x] 멀티턴 대화 생성 로직 구현
  - [x] 기억 저장/활용 시점 시뮬레이션
- [x] 로컬 모델로 에피소드 생성 실행
- [x] 공개 데이터 변환 파이프라인 (multi_session_dialogue, korean_role_playing)

### 1.2 Heuristic Reward 계산기 ✅

- [x] Reward 함수 설계 및 구현 (`src/aimemory/reward/`)
  - [x] (+) 저장한 키워드/엔티티가 미래 턴에 재등장
  - [x] (-) 사용자가 비슷한 질문을 반복 (기억 실패 신호)
  - [x] (+) 저장 기억의 토큰 길이가 짧을수록 (효율 보너스)
- [x] Self-play 결과에 대해 reward 자동 계산 테스트
- [x] Reward 분포 시각화 및 분석 (`scripts/check_distribution.py`)

### 1.3 학습 데이터셋 구성 ✅

- [x] (state, action, reward) 트리플 형식으로 데이터셋 구성 — **2,533,479 행**
  - train: 2,026,182 / val: 254,377 / test: 252,920
- [x] 데이터셋 통계: SAVE 49.3%, SKIP 45.9%, RETRIEVE 4.8%
- [x] 학습/검증/테스트 셋 분리 (`data/splits/{train,val,test}.parquet`)
- [x] Parquet 포맷 확정

### 1.4 API 비용 최적화 검증 ✅

- [x] 로컬 모델 대량 생성 파이프라인 동작 확인
- [x] API 미사용 (100% 로컬 생성 + 공개 데이터)

---

## Step 2: Rule-Based Baseline + Online MLP Bandit ✅ 완료

> **목표**: 대화에서 SAVE/SKIP/RETRIEVE를 결정하는 메모리 정책 구현
> **아키텍처 전환**: Offline DQN → Rule-Based importance scoring + Online MLP Bandit
> **전환 사유**: Offline DQN 학습 실패 (loss 증가, accuracy 하락). 프로덕션 메모리 시스템(ChatGPT/Mem0)이 모두 rule-based + 간단한 online 학습으로 수렴하는 패턴 확인.
> **완료일**: 2026-02-27

### 2.1 Rule-Based Importance Scoring ✅

- [x] `_compute_importance()` 구현 (`src/aimemory/online/policy.py`)
  - 개인정보(+0.4), 선호(+0.35), 기술(+0.3), 감정(+0.2), 키워드 밀도(+0.15) 가중합
  - 고확신 영역 (importance >= 0.7) → 직접 SAVE
  - 저확신 영역 (importance <= 0.1) → 직접 SKIP
  - 중간 영역 → MLP bandit에 위임
- [x] `_should_retrieve()` 구현 (`src/aimemory/online/policy.py`)
  - 질문 패턴 감지 + discourse marker 감지 → 저장된 메모리 있으면 RETRIEVE
- [x] `MemoryPolicyAgent.decide()` 3단계 파이프라인 교체
  - Phase 0: RETRIEVE 판단 (rule-based)
  - Phase 1: 고/저확신 직접 결정 (rule-based)
  - Phase 2: 중간 영역 MLP bandit 결정

### 2.2 Online MLP Bandit (기존 유지) ✅

- [x] `OnlinePolicy` — 10d 입력, 64d hidden, 3-action MLP contextual bandit (`src/aimemory/online/policy.py`)
  - epsilon-greedy action selection
  - single-step SGD update
  - gossip 프로토콜 호환 (get/set_parameters)
  - checkpoint save/load
- [x] `StateEncoder` — 10d hand-crafted features
  - turn_position, memory_count, keyword_count, is_question, has_personal_info,
    has_preference, has_tech, has_emotion, recent_save_count, recent_retrieve_count
- [x] `FeedbackDetector` 연동 — 한국어 피드백 패턴으로 online reward 수집

### 2.3 Config 교체 ✅

- [x] `ExtractorConfig` → `OnlinePolicyConfig` (`src/aimemory/config.py`)
  - MLP bandit 파라미터 (feature_dim, hidden_dim, lr, epsilon)
  - Rule-based 임계값 (save_threshold=0.7, skip_threshold=0.1)
  - Importance 가중치 (personal/preference/tech/emotion/keyword)
  - Retrieval 설정 (retrieve_top_k, st_model)
- [x] `AppConfig.extractor` → `AppConfig.online_policy`

### 2.4 DQN 파이프라인 제거 ✅

- [x] `src/aimemory/extractor/` 패키지 전체 삭제 (DualHeadDQN, OfflineDQNTrainer, DQNPolicy, EmbeddingDataset, EnhancedStateEncoder)
- [x] `scripts/07_precompute_embeddings.py` 삭제 (27GB 임베딩 파이프라인)
- [x] `scripts/08_train_dqn.py` 삭제 (DQN 학습 스크립트)
- [x] `scripts/09_evaluate_dqn.py` 삭제 (DQN 평가 스크립트)
- [x] `tests/test_extractor.py` 삭제 (DQN 관련 테스트)

### 2.5 테스트 검증 ✅

- [x] 기존 `test_online_policy.py` 26개 테스트 전체 통과
- [x] 신규 테스트 추가: TestImportanceScoring, TestRuleBasedDecision, TestShouldRetrieve, TestOnlinePolicyConfig
- [x] `uv run pytest` — 전체 346개 테스트 통과
- [x] 코드베이스 `extractor` 잔여 참조 없음 확인 (grep clean)

### 2.6 DB 저장/검색 통합 ✅

- [x] `GraphMemoryStore` (ChromaDB 벡터DB) 구현 완료 (`src/aimemory/memory/`)
- [x] 기억 단위 스키마 확장
  - [x] 다해상도 텍스트 (Level 0/1/2) — `level1_text`, `level2_text` 필드
  - [x] timestamp — `created_at` (기존) + `access_count` 자동 증가
  - [x] 원본 대화 ID — `conversation_id` 필드
- [x] `active` 필터 (비활성 메모리 검색 제외)
- [x] `pin_memory()` / `unpin_memory()` 망각 보호 메서드
- [x] 저장/검색 end-to-end 통합 테스트 (`TestE2EIntegration`)

### 2.7 불변 규칙 필터 통합 ✅

- [x] `ImmutableMemoryError` + immutable 메모리 수정/삭제 차단
- [x] `core_principle` 카테고리 지원
- [x] 삭제 요청 시 즉시 완전 삭제 (`delete_memory()` + graph edge 정리)
- [x] 원본 대화 ID 참조 (`source_turn_id`)
- [x] Rule hash 교차 검증 (`RuleVerifier`, SHA-256)
- [x] Rule-based 정책과 완전 분리 구조
- ~~비밀번호/API 키 차단~~ (제외)
- ~~의료정보 차단~~ (제외)

---

## Step 3: 망각 + 다해상도 저장 (예상 1~2주) ✅

> **목표**: 기억 품질을 높이는 망각 시스템 + 토큰 효율을 높이는 다해상도 저장 구현
> **성공률**: 90%+
> **성공 기준**: 기억 1000개 상태에서 망각 전/후 검색 정밀도 비교 → 개선 확인

### 3.1 중요도 계산기 ✅

- [x] access_count 추적 시스템 구현 — `search()` 시 자동 증가
- [x] recency_decay 함수 구현 (지수 감쇠: e^(-λ × days)) — `ImportanceCalculator.recency_decay()`
- [x] importance = (1 + access_count) × recency_decay + related_boost × len(related_ids) — `ImportanceCalculator.compute()`
- [x] 테스트 검증 (`tests/test_forgetting.py::TestImportanceCalculator`)

### 3.2 망각 단계 구현 ✅

- [x] **압축 단계**: importance < threshold_compress
  - [x] Level 2 entity triple로 압축 (`compress_memory()`)
  - [x] 원본 대화 ID (`conversation_id`)로 재구성 가능
- [x] **비활성화 단계**: importance < threshold_deactivate
  - [x] `active=false` 설정, 검색에서 제외
  - [x] `include_inactive=True`로 복원 가능
- [x] **영구 삭제 단계**: 비활성화 후 deactivation_days 경과
  - [x] 완전 삭제 + `AuditEntry` 기록
- [x] 임계값 조정 → 각 단계 전이 테스트 (`tests/test_forgetting.py::TestForgettingPipeline`)

### 3.3 망각 방지 메커니즘 ✅

- [x] 사용자 핀 기능 구현 — `pin_memory()` / `unpin_memory()`
- [x] 연결된 기억 보호 — `related_boost * len(related_ids)` 중요도 보너스
- [x] immutable/pinned 스킵 — `ForgettingPipeline.run()` 에서 자동 보호

### 3.4 다해상도 텍스트 시스템 ✅

- [x] Level 0: 원문 그대로 저장 (`content`)
- [x] Level 1: keyword 포함 문장 추출 (1-2문장, max 100자) — `generate_level1()`
- [x] Level 2: entity triple 추출 (subject,predicate,object) — `generate_level2()`
- [x] `generate_all_levels()` → `MultiResolutionText` 반환
- [x] 토큰 추정: `estimate_tokens()` (len/2.5 근사)
- [x] 테스트 17개 통과 (`tests/test_resolution.py`)

### 3.5 토큰 Budget 기반 해상도 선택기 ✅

- [x] `ContextComposer` 구현 — relevance 정렬 → greedy Level 선택
- [x] `compose()` — token_budget 내 최적 해상도 자동 선택
- [x] `format_context()` — `[L0]/[L1]/[L2]` 태그 포맷
- [x] Top-K 기억 선택 + 해상도 조합 최적화
- [x] 테스트 8개 통과 (`tests/test_composer.py`)

---

## Step 4: 수면 사이클 (Sleep Cycle) + 기억 통합 ✅ 완료

> **목표**: 주기적 배치 프로세스로 기억 유지보수 자동화
> **완료일**: 2026-02-27
> **설계 결정**: Progressive 차원 확장(64→128→256)은 삭제된 DQN 시스템용이었으므로 제외. 768d SentenceTransformer + 10d hand-crafted 상태 피처 사용.

### 4.1 기억 통합 (Consolidation) ✅

- [x] `MemoryConsolidator`: 시맨틱 유사도 기반 중복 탐지 + 병합
- [x] `MergeRecord`, `ConsolidationResult`: 병합 감사 기록
- [x] immutable 보호, access_count 기반 생존자 결정
- [x] 키워드/related_ids 합집합 보존

### 4.2 수면 사이클 러너 ✅

- [x] `SleepCycleRunner`: 4개 태스크 순차 실행 (통합→해상도→망각→체크포인트)
- [x] `SleepCycleReport`: 실행 결과 + to_dict() + summary()
- [x] 태스크 간 에러 격리 (하나 실패해도 나머지 실행)
- [x] JSON 리포트 저장

### 4.3 CLI + 설정 ✅

- [x] `scripts/07_sleep_cycle.py`: CLI 진입점 (argparse + dry-run)
- [x] `SleepCycleConfig`: 통합 설정 (AppConfig에 추가)
- [x] 다해상도 텍스트 재생성 (level1/level2 누락 메모리)
- [x] RL 모델 checkpoint 저장

### 4.4 테스트 ✅

- [x] 17개 테스트 추가 (총 409개 통과)

---

## Step 5: RL Re-ranker ✅ 완료

> **목표**: DB 검색 결과를 사용자 패턴에 맞게 재정렬하는 경량 RL
> **완료일**: 2026-02-28

### 5.1 Re-ranker 모델 ✅

- [x] Pointwise MLP (8d→32d→1d, 321 params) 설계 및 구현 (`src/aimemory/online/reranker.py`)
- [x] 8d 피처: chroma_similarity, keyword_overlap, category_match, recency, access_frequency, content_length_ratio, has_related, resolution_available
- [x] top-10 검색 → re-rank → top-3 선택 파이프라인 (`ReRanker` facade)
- [x] Online SGD 학습 (FeedbackDetector 연동)
- [x] `MemoryPolicyAgent` 통합 (reranker 옵션)
- [x] `ReRankerConfig` 설정 (AppConfig에 추가)

### 5.2 A/B 비교 프레임워크 ✅

- [x] `ABComparator`: baseline vs re-ranked 비교 (`src/aimemory/online/ab_comparator.py`)
- [x] `ABResult`, `ABReport`: overlap, position 변화, p50/p95 레이턴시
- [x] Graceful degradation (disabled/latency exceeded → ChromaDB fallback)

### 5.3 테스트 ✅

- [x] 38개 테스트 추가 (test_reranker.py 30개 + test_ab_comparator.py 8개)

---

## Step 6: P2P Federated Learning ✅ 완료

> **목표**: 여러 사용자의 학습 결과를 P2P로 공유하여 개별 학습 속도 향상
> **완료일**: 2026-02-28

### 6.1 모델 직렬화 ✅

- [x] RL 모델 serialize/deserialize 구현 (`OnlinePolicy.get_parameters() / set_parameters()`)
- [x] checkpoint save/load 구현 (`OnlinePolicy.save_checkpoint() / load_checkpoint()`)

### 6.2 P2P 네트워크 ✅

- [x] Gossip 프로토콜 구현 (`src/aimemory/online/gossip.py`)
- [x] Online Policy 구현 (`src/aimemory/online/policy.py`)
- [x] TCP Transport 구현 (`src/aimemory/online/transport.py`) — asyncio 기반, stdlib only
- [x] 노드 2개 간 실제 네트워크 통신 테스트

### 6.3 보안 ✅

- [x] Differential Privacy — Gaussian mechanism, L2 norm clipping (`gossip.py`)
- [x] Byzantine-resilient Aggregation — Krum 알고리즘
- [x] Rule Hash 검증 (`src/aimemory/online/rule_verifier.py`) — SHA-256 of SecurityConfig
- [x] 규칙 변조 노드 → `_rejected_peers`에 추가, 네트워크 접속 거부

### 6.4 검증 데이터셋 ✅

- [x] 한국어 대화 검증 데이터셋 다운로드 (`data/raw/public/validation/`)

### 6.5 벤치마크 ✅

- [x] `scripts/08_convergence_benchmark.py`: 개별 vs P2P 수렴 속도 비교

### 6.6 테스트 ✅

- [x] 34개 테스트 추가 (gossip DP 8개 + rule_verifier 8개 + transport 3개 등)

---

## Step 7: MCP 서버 패키징 및 출시 ✅ 완료

> **목표**: 개발 완료된 시스템을 MCP 서버로 패키징하여 배포
> **완료일**: 2026-02-28

### 7.1 MCP 서버 구현 ✅

- [x] FastMCP 서버 (`src/aimemory/mcp/server.py`) — 12개 tool 정의
- [x] `MemoryBridge` 오케스트레이터 (`src/aimemory/mcp/bridge.py`)
- [x] 도구: memory_save, memory_search, auto_search, memory_update, memory_delete, memory_get_related, memory_pin, memory_unpin, memory_stats, sleep_cycle_run, policy_status, policy_decide
- [x] `auto_search`: 매 턴 관련 기억 자동 검색 + multi-resolution context 조합

### 7.2 통합 테스트 ✅

- [x] E2E 시나리오 테스트 — 10회 연속 대화에서 기억 자동 활용
- [x] 세션 간 persistence 테스트
- [x] MCP 프로토콜 통합 테스트

### 7.3 배포 준비 ✅

- [x] `mcp[cli]>=1.2.0` 의존성 추가 (pyproject.toml)
- [x] `aimemory-mcp` 스크립트 entry point
- [x] `MCPServerConfig` 설정 (AppConfig에 추가)
- [x] Claude Desktop 설정 지원 (`python -m aimemory.mcp` stdio transport)

### 7.4 테스트 ✅

- [x] 36개 테스트 추가 (bridge 27개 + server 7개 + e2e 2개)

---

## Step 8: RL Evolution + GraphRAG Integration ✅ 완료

> **목표**: RL의 제한적 역할(10d 피처 + 771-param MLP → 규칙 70% 지배)을 해소하고, ChromaDB cosine 한계(관계 기반 추론 불가)를 GraphRAG로 보완
> **설계 원칙**: Additive (기존 클래스 수정 최소화), Opt-in config flags, 기존 테스트 무손상
> **완료일**: 2026-02-28

### 8.1 Experience Replay Buffer ✅

- [x] `ReplayBuffer`: circular deque (capacity=5000) + batch sampling (`src/aimemory/online/replay_buffer.py`)
- [x] `Experience` namedtuple: (state, action, reward, next_state)
- [x] save/load 직렬화
- [x] 9개 테스트

### 8.2 Enhanced State Encoder ✅

- [x] `EnhancedStateEncoder`: 768d SentenceTransformer embedding + 10d hand-crafted → 778d (`src/aimemory/online/enhanced_encoder.py`)
- [x] `set_embedding_fn(fn)`: 외부 임베딩 함수 주입 (GraphMemoryStore와 모델 공유)
- [x] 13개 테스트

### 8.3 Progressive Autonomy ✅

- [x] `ProgressiveAutonomy`: confidence 기반 임계값 완화 (`src/aimemory/online/autonomy.py`)
- [x] 긍정 피드백 누적 → save_threshold ↓, skip_threshold ↑, RL zone 확장 (60% → 최대 90%)
- [x] 부정 피드백 → confidence 대폭 감소 (안전장치)
- [x] `OnlinePolicyConfig`에 `use_enhanced_policy`, `use_progressive_autonomy`, `autonomy_confidence_threshold` 추가
- [x] 11개 테스트

### 8.4 KnowledgeGraph ✅

- [x] `KnowledgeGraph`: networkx DiGraph 기반 지식 그래프 (`src/aimemory/memory/knowledge_graph.py`)
- [x] level2_text CSV("subject,predicate,object") 자동 파싱 → 트리플 추가
- [x] 다중 hop 탐색, 경로 쿼리, 엔티티 컨텍스트, store 기반 전체 재구축
- [x] 18개 테스트

### 8.5 Implicit Reward Detector ✅

- [x] `ImplicitRewardDetector`: 대화 흐름 기반 암묵적 보상 (`src/aimemory/reward/implicit_detector.py`)
- [x] 대화 지속(+0.3), 화제 확장(+0.2), 짧은 응답 종료(-0.1)
- [x] 5개 테스트

### 8.6 Enhanced Online Policy ✅

- [x] `EnhancedOnlinePolicy(OnlinePolicy)`: drop-in replacement (`src/aimemory/online/enhanced_policy.py`)
- [x] `_EnhancedMLP`: 778d → 256 → 128 → 3 (~233k params, dropout)
- [x] Experience replay + batch SGD + progressive autonomy 연동
- [x] 기존 `OnlinePolicy` 인터페이스 100% 호환
- [x] 10개 테스트

### 8.7 GraphRAG Hybrid Retrieval ✅

- [x] `GraphRetriever`: ChromaDB vector + KnowledgeGraph traversal 하이브리드 검색 (`src/aimemory/memory/graph_retriever.py`)
- [x] vector_weight(0.6) + graph_weight(0.4) 점수 융합
- [x] 한국어 명사 + 기술 용어 엔티티 추출
- [x] 부정 관계(싫어함 등) 감지 → 부정 컨텍스트 포함
- [x] 8개 테스트

### 8.8 Graph-Aware ReRanker ✅

- [x] `ReRankFeatureExtractor` 확장: 8d → 11d (기존 8d + graph 3d) (`src/aimemory/online/reranker.py` MODIFY)
- [x] 신규 피처: graph_connection_count, graph_hop_distance, has_negative_relation
- [x] KG 미주입 시 3개 피처 = 0.0 (backward compatible)
- [x] `ReRankerConfig`에 `feature_dim=11`, `use_graph_features` 추가
- [x] 4개 테스트 추가 (기존 30개 유지)

### 8.9 KG Auto-Builder ✅

- [x] `GraphMemoryStore.__init__`에 `knowledge_graph` 파라미터 추가 (`src/aimemory/memory/graph_store.py` MODIFY)
- [x] `add_memory()`: level2_text 있으면 KG에 자동 트리플 추가
- [x] `delete_memory()`: KG에서 관련 트리플 자동 제거
- [x] 초기 로딩 시 `kg.rebuild_from_store(self)` 호출
- [x] 3개 테스트 추가

### 8.10 Bridge Integration ✅

- [x] `MemoryBridge`에 `use_enhanced_policy`, `use_graph_rag` 파라미터 추가 (`src/aimemory/mcp/bridge.py` MODIFY)
- [x] 플래그 우선순위: 파라미터 > 환경변수 (`AIMEMORY_ENHANCED_POLICY=1`, `AIMEMORY_GRAPH_RAG=1`) > config
- [x] Enhanced 모드: `EnhancedOnlinePolicy` + replay buffer + autonomy
- [x] GraphRAG 모드: `KnowledgeGraph` + `GraphRetriever` + graph-aware ReRanker
- [x] 기본값 False → 기존 동작 100% 보존
- [x] `MCPServerConfig`에 `use_enhanced_policy`, `use_graph_rag` 추가
- [x] 6개 테스트 추가

### 8.11 E2E Integration Tests ✅

- [x] Enhanced policy 10-conversation 시뮬레이션
- [x] GraphRAG 관계 추론 ("봉골레 싫어함" 반영)
- [x] Progressive autonomy 임계값 완화 검증
- [x] Legacy mode 무손상 검증
- [x] 4개 테스트

### 8.12 테스트 요약

- [x] 신규 58개 테스트 추가
- [x] 기존 553개 테스트 무손상
- [x] **총 611개 테스트 통과**

---

## 전체 로드맵 요약

| Step | 내용 | 기간 | 성공률 | 상태 |
|------|------|------|--------|------|
| 1 | 학습 데이터 확보 | 2~3주 | ~100% | ✅ 완료 (2026-02-26) |
| 2 | Rule-Based + Online MLP Bandit | 1주 | ~100% | ✅ 완료 (2026-02-27) |
| 3 | 망각 + 다해상도 저장 | 1~2주 | 90%+ | ✅ 완료 (2026-02-27) |
| 4 | 수면 사이클 + 기억 통합 | 1~2주 | 90%+ | ✅ 완료 (2026-02-27) |
| 5 | RL Re-ranker | 1~2주 | 80%+ | ✅ 완료 (2026-02-28) |
| 6 | P2P Federated Learning | 3~4주 | 60~70% | ✅ 완료 (2026-02-28) |
| 7 | MCP 서버 패키징 및 출시 | 1주 | ~100% | ✅ 완료 (2026-02-28) |
| 8 | RL Evolution + GraphRAG | 1일 | ~100% | ✅ 완료 (2026-02-28) |

> **전체 완료!** 8개 Step 모두 구현 완료 (611개 테스트 통과)

---

## 핵심 원칙

- 각 Step은 이전 Step의 결과물을 기반으로 하며, 독립적으로 검증 가능
- **"저장 품질 > 검색 알고리즘"** 이 시스템의 기본 철학
- RL 모델을 serialize 가능하게 설계 → P2P 확장 용이
- MCP 서버는 최종 배포 형태 → 모든 기능 개발 완료 후 패키징
- Feature ≠ Fact: feature vector는 검색 키로만 사용, 저장되는 원본은 fact/텍스트
