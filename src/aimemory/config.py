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
        "r5_speech_act_weight": 0.7,
        "r6_self_reference": 0.7,
        "r7_info_density": 0.5,
        "r8_preference_constraint": 0.9,
        "r9_emotional_salience": 0.4,
        "r10_topic_boundary": 0.6,
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


class ExtractorConfig(BaseModel):
    """DQN feature extractor configuration."""

    # Model architecture
    emb_dim: int = 768
    proj_dim: int = 128
    hand_dim: int = 10
    trunk_dim: int = 128
    n_actions: int = 3
    feature_dim: int = 64
    dropout: float = 0.1

    # Training hyperparameters
    batch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.99
    target_sync: int = 1000
    max_epochs: int = 20
    patience: int = 3
    max_grad_norm: float = 1.0
    epsilon: float = 0.1

    # Class weights for imbalanced actions
    class_weights: dict[int, float] = Field(
        default_factory=lambda: {0: 1.0, 1: 0.7, 2: 3.0}
    )

    # Sentence-transformer model
    st_model: str = "jhgan/ko-sroberta-multitask"


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


class AppConfig(BaseModel):
    """Top-level application configuration."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    selfplay: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    paths: DataPaths = Field(default_factory=DataPaths)
    num_episodes: int = 1000
