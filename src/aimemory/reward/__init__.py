"""Reward module for the AI Memory System."""

from .calculator import RewardCalculator
from .signals import (
    compute_r1_keyword_reappearance,
    compute_r10_topic_boundary,
    compute_r11_user_feedback,
    compute_r2_repeated_question_penalty,
    compute_r3_efficiency,
    compute_r4_retrieval_relevance,
    compute_r5_speech_act_weight,
    compute_r6_self_reference,
    compute_r7_info_density,
    compute_r8_preference_constraint,
    compute_r9_emotional_salience,
)

__all__ = [
    "RewardCalculator",
    "compute_r1_keyword_reappearance",
    "compute_r2_repeated_question_penalty",
    "compute_r3_efficiency",
    "compute_r4_retrieval_relevance",
    "compute_r5_speech_act_weight",
    "compute_r6_self_reference",
    "compute_r7_info_density",
    "compute_r8_preference_constraint",
    "compute_r9_emotional_salience",
    "compute_r10_topic_boundary",
    "compute_r11_user_feedback",
]
