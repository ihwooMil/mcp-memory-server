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

### 2.7 불변 규칙 필터 통합

- [ ] Level 1 하드코딩 규칙 구현 (별도 후속 작업)
  - [ ] 비밀번호/API 키 저장 차단
  - [ ] 개인 의료정보 암호화 없이 저장 차단
  - [ ] 삭제 요청 시 즉시 완전 삭제
  - [ ] 원본 대화 ID 참조 필수
- [ ] API 키/비밀번호 포함 대화 → 저장 차단 테스트
- [ ] Rule-based 정책과 완전 분리 구조 확인

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

## Step 4: 수면 사이클 + Progressive 확장 (예상 1~2주)

> **목표**: 주기적 배치 프로세스로 feature 재인코딩, 차원 확장, 기억 통합, 망각 통합 실행
> **성공률**: 90%+
> **성공 기준**: 수면 사이클 1회 실행 후 성능 저하 없음 + 검색 정밀도 유지/향상

### 4.1 수면 사이클 스케줄러

- [ ] 배치 스크립트/cron 스케줄러 구현
- [ ] 수동 트리거 지원
- [ ] 실행 로그 및 리포트 생성

### 4.2 필수 작업 구현

- [ ] **Feature 재인코딩**
  - [ ] 0 패딩된 기억의 원본 fact/텍스트를 현재 RL encoder로 재추출
  - [ ] 수면 사이클 후 0패딩 feature 수 = 0 확인
- [ ] **차원 확장 판단**
  - [ ] 기억 수 기준 확장 트리거 (~2K→64→128, ~10K→128→256)
  - [ ] 검색 precision 임계값 기반 트리거
- [ ] **Progressive 차원 확장 (64→128)**
  - [ ] 기존 feature zero-padding 구현
  - [ ] 0패딩 상태에서도 검색 동작 확인
  - [ ] 재인코딩 후 정밀도 향상 확인
- [ ] **기억 통합 (Consolidation)**
  - [ ] 중복 기억 merge 로직
  - [ ] Diff 저장 적용
  - [ ] 중복 기억 5쌍 입력 → 통합 후 건수 감소 확인
- [ ] **망각 처리 통합**
  - [ ] Step 3의 망각 시스템을 수면 사이클에 통합

### 4.3 선택 작업

- [ ] 다해상도 텍스트 재생성
- [ ] RL 모델 checkpoint 저장 (복원점 확보)

### 4.4 Online 학습 통합 (선택)

- [ ] 실시간 reward 수집 파이프라인
- [ ] Experience Replay Buffer 구현
- [ ] Catastrophic Forgetting 방지 (EWC 또는 Replay)
- [ ] Delayed Reward 처리 (주기적 활용도 평가 loop)
- [ ] Policy update → 추출 품질 변화 측정

---

## Step 5: RL Re-ranker (예상 1~2주, 선택)

> **목표**: DB 검색 결과를 사용자 패턴에 맞게 재정렬하는 경량 RL
> **성공률**: 80%+
> **성공 기준**: re-rank 후 응답 품질 측정 가능하게 개선, 레이턴시 증가 30ms 이하

### 5.1 Re-ranker 모델

- [ ] 경량 RL Re-ranker 설계 및 구현
- [ ] top-10 검색 결과 → re-rank → top-3 선택 파이프라인
- [ ] 학습 데이터 수집 (사용자 검색 패턴 기반)
- [ ] Online 학습 지원

### 5.2 성능 검증

- [ ] 레이턴시 측정: DB 검색 ~10ms + re-ranker ≤ 20ms (총 ≤ 30ms)
- [ ] A/B 비교: re-ranker 적용 전/후 응답 품질 비교
- [ ] re-rank 전/후 검색 결과 적절성 평가

---

## Step 6: P2P Federated Learning (예상 3~4주, 선택) 🔄 일부 구현

> **목표**: 여러 사용자의 학습 결과를 P2P로 공유하여 개별 학습 속도 향상
> **성공률**: 60~70%
> **성공 기준**: 노드 2개 간 gradient 공유 → 개별 학습 대비 수렴 속도 향상

### 6.1 모델 직렬화 ✅

- [x] RL 모델 serialize/deserialize 구현 (`OnlinePolicy.get_parameters() / set_parameters()`)
- [x] checkpoint save/load 구현 (`OnlinePolicy.save_checkpoint() / load_checkpoint()`)

### 6.2 P2P 네트워크 ✅ (프로토타입)

- [x] Gossip 프로토콜 구현 (`src/aimemory/online/gossip.py`)
  - [x] 노드 간 파라미터 교환
  - [x] 글로벌/로컬 레이어 분리 (gossip 인터페이스 호환)
- [x] Online Policy 구현 (`src/aimemory/online/policy.py`)
  - [x] 실시간 학습 + gossip 통합
  - [x] StateEncoder (10d hand-crafted features)
- [ ] 노드 2개 간 실제 네트워크 통신 테스트

### 6.3 보안 ✅ (일부)

- [ ] Differential Privacy 적용 (gradient 노이즈)
- [x] Byzantine-resilient Aggregation — **Krum 알고리즘** 구현
- [ ] Level 3 규칙 검증 (불변 규칙 해시 교차 검증)
- [ ] 규칙 변조 노드 → 네트워크 접속 거부 테스트

### 6.4 검증 데이터셋 확보 ✅

- [x] 한국어 대화 검증 데이터셋 다운로드 (`data/raw/public/validation/`)
  - chatbot_data.csv (11,823건, songys/Chatbot_data)
  - ko_wikidata_qa.jsonl (10,000건, maywell/ko_wikidata_QA)
  - kullm_v2.jsonl (10,000건, nlpai-lab/kullm-v2)

### 6.5 성능 검증

- [ ] 개별 학습 vs P2P 학습 수렴 속도 비교
- [ ] 프라이버시 검증: gradient에서 원본 데이터 역추적 불가 테스트

---

## Step 7: MCP 서버 패키징 및 출시 (예상 1주)

> **목표**: 개발 완료된 시스템을 MCP 서버로 패키징하여 배포
> **성공 기준**: 10회 연속 대화에서 이전 대화의 기억이 자동으로 활용, 사용자 개입 없이 동작

### 7.1 MCP 서버 구현

- [ ] MCP 서버 코드 작성 (tool 정의 + 서버 실행)
- [ ] MCP Inspector에서 모든 tool 호출 성공 확인
- [ ] 자동 검색 미들웨어 구현 (매 턴 관련 기억 자동 삽입)

### 7.2 통합 테스트

- [ ] 사용자가 기억 검색 요청 없이도 관련 기억이 응답에 반영되는지 확인
- [ ] End-to-end 시나리오 테스트 (10회 연속 대화)
- [ ] 수면 사이클 자동화 (cron 설정) → 24시간 내 자동 실행 확인

### 7.3 배포 준비

- [ ] 설치 가이드 + README 작성
- [ ] README 따라 설치 → Claude Desktop 정상 동작 확인
- [ ] 의존성 정리 및 패키지 매니저 설정
- [ ] 에러 핸들링 및 로깅 최종 점검

---

## 전체 로드맵 요약

| Step | 내용 | 기간 | 성공률 | 상태 |
|------|------|------|--------|------|
| 1 | 학습 데이터 확보 | 2~3주 | ~100% | ✅ 완료 (2026-02-26) |
| 2 | Rule-Based + Online MLP Bandit | 1주 | ~100% | ✅ 완료 (2026-02-27) |
| 3 | 망각 + 다해상도 저장 | 1~2주 | 90%+ | ⬜ 미착수 |
| 4 | 수면 사이클 + Progressive 확장 | 1~2주 | 90%+ | ⬜ 미착수 |
| 5 | RL Re-ranker (선택) | 1~2주 | 80%+ | ⬜ 미착수 |
| 6 | P2P Federated Learning (선택) | 3~4주 | 60~70% | 🔄 일부 구현 (gossip+Krum) |
| 7 | MCP 서버 패키징 및 출시 | 1주 | ~100% | ⬜ 미착수 |

> **총 예상 기간**: 필수(Step 1~4 + 7) = 7~12주 / 전체(선택 포함) = 11~18주

---

## 핵심 원칙

- 각 Step은 이전 Step의 결과물을 기반으로 하며, 독립적으로 검증 가능
- **"저장 품질 > 검색 알고리즘"** 이 시스템의 기본 철학
- RL 모델을 serialize 가능하게 설계 → P2P 확장 용이
- MCP 서버는 최종 배포 형태 → 모든 기능 개발 완료 후 패키징
- Feature ≠ Fact: feature vector는 검색 키로만 사용, 저장되는 원본은 fact/텍스트
