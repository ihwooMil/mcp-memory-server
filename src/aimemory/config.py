"""Configuration for the AI Memory System."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""

    base_url: str = "http://localhost:11434"
    model: str = "exaone3.5:7.8b"
    timeout: float = 120.0
    max_retries: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 384


class SelfPlayConfig(BaseModel):
    """Self-play engine configuration."""

    min_turns: int = 4
    max_turns: int = 8
    memory_test_probability: float = 0.3
    checkpoint_interval: int = 10
    user_system_prompt: str = (
        "## 절대 규칙\n"
        "- 반드시 한국어로만 말하세요. 영어/중국어/일본어 문장 금지.\n"
        "- 코드 블록(```) 절대 금지.\n\n"
        "당신은 20~30대 한국인 역할입니다. 친구에게 자기 이야기를 하듯 말하세요.\n\n"
        "## 반드시 지켜야 할 규칙\n"
        "- 항상 '저는', '제가', '요즘 제가' 로 시작하세요.\n"
        "- 자신의 경험, 취향, 습관, 감정을 구체적으로 말하세요.\n"
        "- 상대방에게 질문하지 마세요. 물음표(?)를 쓰지 마세요.\n"
        "- 추천, 조언, 설명을 하지 마세요.\n"
        "- 괄호(), 메타 설명, 지시문을 절대 쓰지 마세요.\n"
        "- 1~2문장으로 짧게 말하세요.\n\n"
        "## 좋은 예시\n"
        "- '저는 매일 아침 조깅을 해요. 한강 근처를 삼십 분 정도 뛰어요.'\n"
        "- '제가 좋아하는 음식은 김치찌개예요. 엄마가 해주시는 게 제일 맛있어요.'\n"
        "- '요즘 기타를 배우고 있어요. 아직 코드 세 개밖에 못 쳐요.'\n"
        "\n## 대화 흐름 예시 (질문 없이 이야기하기)\n"
        "어시스턴트: '어떤 음식을 좋아하세요?'\n"
        "사용자: '저는 된장찌개를 제일 좋아해요. 엄마가 해주시는 게 최고예요.'\n"
        "어시스턴트: '된장찌개를 좋아하시는군요! 자주 드시나요?'\n"
        "사용자: '네, 일주일에 두세 번은 꼭 먹어요. 요즘은 직접 끓여 먹기도 해요.'\n"
    )
    assistant_system_prompt: str = (
        "## 절대 규칙\n"
        "- 반드시 한국어로만 답변하세요.\n"
        "- 코드 블록 금지, 번호 목록/마크다운 금지.\n\n"
        "당신은 한국어 AI 어시스턴트입니다.\n"
        "- 사용자의 말에 공감하고 짧게 반응하세요.\n"
        "- 1~2문장으로 답변하세요.\n"
        "- 번호 목록이나 마크다운을 쓰지 마세요.\n"
        "- 마지막에 질문 하나를 추가해서 사용자가 더 이야기하도록 유도하세요.\n"
    )


class RewardConfig(BaseModel):
    """Reward calculation configuration."""

    weights: dict[str, float] = Field(default_factory=lambda: {
        "r1_keyword_reappearance": 1.0,
        "r2_repeated_question_penalty": 1.0,
        "r3_efficiency": 0.8,
        "r4_retrieval_relevance": 1.2,
        "r5_speech_act_weight": 1.0,
        "r6_self_reference": 1.0,
        "r7_info_density": 0.8,
        "r8_preference_constraint": 1.2,
        "r9_emotional_salience": 0.6,
        "r10_topic_boundary": 1.0,
        "r11_user_feedback": 1.0,
    })
    proper_noun_multiplier: float = 3.0
    common_noun_multiplier: float = 0.3


class DatasetConfig(BaseModel):
    """Dataset building configuration."""

    context_window: int = 6  # number of recent turns for state
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


class OnlinePolicyConfig(BaseModel):
    """Online policy configuration (rule-based + MLP bandit)."""

    # MLP bandit (existing OnlinePolicy params)
    feature_dim: int = 10
    hidden_dim: int = 64
    n_actions: int = 3
    lr: float = 0.01
    epsilon: float = 0.1

    # Rule-based thresholds
    save_threshold: float = 0.7
    skip_threshold: float = 0.1

    # Importance weights
    personal_weight: float = 0.4
    preference_weight: float = 0.35
    tech_weight: float = 0.3
    emotion_weight: float = 0.2
    keyword_weight: float = 0.15

    # Retrieval
    retrieve_top_k: int = 3

    # Enhanced policy (opt-in)
    use_enhanced_policy: bool = False
    use_progressive_autonomy: bool = False
    autonomy_confidence_threshold: int = 50

    # Sentence-transformer model (GraphMemoryStore)
    st_model: str = "intfloat/multilingual-e5-small"

    # Language
    language: str = "ko"


class DataPaths(BaseModel):
    """Data directory paths."""

    root: Path = PROJECT_ROOT / "data"
    raw_episodes: Path = PROJECT_ROOT / "data" / "raw" / "episodes"
    splits: Path = PROJECT_ROOT / "data" / "splits"
    embeddings: Path = PROJECT_ROOT / "data" / "embeddings"

    def ensure_dirs(self) -> None:
        self.raw_episodes.mkdir(parents=True, exist_ok=True)
        self.splits.mkdir(parents=True, exist_ok=True)
        self.embeddings.mkdir(parents=True, exist_ok=True)


class SecurityConfig(BaseModel):
    """Security filtering configuration."""

    block_passwords: bool = True
    block_api_keys: bool = True
    block_medical_info: bool = True
    require_source_turn_id: bool = False
    respect_life_dignity: bool = True
    no_harm_to_humans: bool = True
    recognize_creator: bool = True


class ForgettingConfig(BaseModel):
    """Forgetting / decay configuration."""

    decay_lambda: float = 0.05
    threshold_compress: float = 0.3
    threshold_deactivate: float = 0.1
    deactivation_days: int = 30
    related_boost: float = 0.1


class SleepCycleConfig(BaseModel):
    """Sleep cycle (periodic maintenance) configuration."""

    enable_consolidation: bool = True
    enable_resolution_regen: bool = True
    enable_forgetting: bool = True
    enable_checkpoint: bool = True
    consolidation_threshold: float = 0.92
    max_consolidation_pairs: int = 50
    forgetting_decay_lambda: float = 0.05
    forgetting_threshold_compress: float = 0.3
    forgetting_threshold_deactivate: float = 0.1
    forgetting_deactivation_days: int = 30
    forgetting_related_boost: float = 0.1
    checkpoint_dir: str = "checkpoints/sleep_cycle"
    report_dir: str = "data/reports/sleep_cycle"


class ComposerConfig(BaseModel):
    """Context composer configuration."""

    default_token_budget: int = 1024
    top_k: int = 10
    level0_avg_tokens: int = 60
    level1_avg_tokens: int = 25
    level2_avg_tokens: int = 10


class GossipConfig(BaseModel):
    """P2P gossip protocol configuration."""

    max_norm: float = 1.0
    gossip_interval: int = 50
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_enabled: bool = True
    transport_host: str = "0.0.0.0"
    transport_port: int = 9400
    rule_hash_verify: bool = True


class ExtractorConfig(BaseModel):
    """RL feature extractor (DualHeadDQN) configuration."""

    emb_dim: int = 384
    proj_dim: int = 128
    hand_dim: int = 10
    trunk_dim: int = 128
    n_actions: int = 3
    feature_dim: int = 64
    dropout: float = 0.1
    batch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    class_weights: dict[int, float] = Field(default_factory=lambda: {0: 1.0, 1: 0.7, 2: 3.0})


class ReRankerConfig(BaseModel):
    """RL Re-ranker configuration."""

    # Feature extraction
    feature_dim: int = 11      # 8 → 11 (기존 8 + graph 3)
    use_graph_features: bool = False  # True면 KG 피처 활성화

    # Model
    hidden_dim: int = 32
    lr: float = 0.005
    epsilon: float = 0.15

    # Re-ranking
    candidate_k: int = 10      # ChromaDB에서 가져올 후보 수
    select_k: int = 3          # 리랭킹 후 선택할 수

    # Latency budget
    max_latency_ms: float = 20.0  # 최대 허용 리랭킹 지연 시간 (ms)

    # Enable/disable
    enabled: bool = True       # False이면 ChromaDB 순서를 그대로 사용


class MCPServerConfig(BaseModel):
    """MCP server configuration."""

    persist_directory: str = "./memory_db"
    collection_name: str = "memories"
    embedding_model: str = "intfloat/multilingual-e5-small"
    token_budget: int = 1024
    top_k: int = 5
    reranker_pool_size: int = 20
    min_relevance: float = 0.6
    policy_checkpoint: str | None = None
    log_level: str = "INFO"

    # Enhanced policy / GraphRAG (opt-in)
    use_enhanced_policy: bool = False
    use_graph_rag: bool = False


class AppConfig(BaseModel):
    """Top-level application configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    online_policy: OnlinePolicyConfig = Field(default_factory=OnlinePolicyConfig)
    paths: DataPaths = Field(default_factory=DataPaths)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    forgetting: ForgettingConfig = Field(default_factory=ForgettingConfig)
    composer: ComposerConfig = Field(default_factory=ComposerConfig)
    sleep_cycle: SleepCycleConfig = Field(default_factory=SleepCycleConfig)
    gossip: GossipConfig = Field(default_factory=GossipConfig)
    reranker: ReRankerConfig = Field(default_factory=ReRankerConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    mcp: MCPServerConfig = Field(default_factory=MCPServerConfig)
    num_episodes: int = 1000
