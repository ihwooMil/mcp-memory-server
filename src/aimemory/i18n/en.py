"""English language patterns for AIMemory."""

from aimemory.i18n import LanguagePatterns, register

_PATTERNS = LanguagePatterns(
    # ── feedback_detector: positive memory patterns ──
    positive_feedback=[
        (r"(?i)\b(?:that'?s?\s+)?(?:right|correct|exactly)\b", 0.95, "right/correct/exactly"),
        (r"(?i)\byes,?\s+(?:that'?s?\s+)?(?:right|correct|it)\b", 0.95, "yes that's right"),
        (r"(?i)\b(?:good|great|excellent)\s+memory\b", 0.90, "good memory"),
        (r"(?i)\byou\s+remember(?:ed)?\b", 0.90, "you remember(ed)"),
        (r"(?i)\b(?:spot[\s-]?on|on[\s-]?point|nailed\s+it)\b", 0.90, "spot on/nailed it"),
        (r"(?i)\bperfect(?:ly)?\b", 0.85, "perfect"),
        (r"(?i)\b(?:oh\s+)?(?:yeah|yep|yup)\b.*\bthat\b", 0.80, "yeah that"),
        (r"(?i)\bI\s+remember\s+(?:that|now)\b", 0.85, "I remember that/now"),
        (
            r"(?i)\bthat'?s?\s+(?:what|exactly\s+what)\s+I\s+(?:said|meant)\b",
            0.90,
            "that's what I said",
        ),
        (r"(?i)\byou'?(?:re|ve)\s+(?:got|right)\b", 0.85, "you're right/you've got it"),
    ],
    # ── feedback_detector: negative memory patterns ──
    negative_feedback=[
        (r"(?i)\b(?:no|nope),?\s+that'?s?\s+(?:not|wrong)\b", 0.90, "no that's not/wrong"),
        (r"(?i)\bI\s+(?:never|didn'?t)\s+(?:said?|told?|mention)\b", 0.90, "I never said"),
        (r"(?i)\bthat'?s?\s+(?:not\s+)?(?:wrong|incorrect)\b", 0.85, "that's wrong/incorrect"),
        (r"(?i)\byou'?(?:re|ve)?\s+(?:got\s+(?:it|that)\s+)?wrong\b", 0.85, "you're wrong"),
        (r"(?i)\bI\s+already\s+(?:told|said|mentioned)\b", 0.90, "I already told you"),
        (
            r"(?i)\byou\s+(?:already\s+)?asked\s+(?:me\s+)?(?:that|this)\b",
            0.85,
            "you already asked",
        ),
        (r"(?i)\b(?:forgot|forgotten|don'?t\s+remember)\b", 0.85, "forgot/don't remember"),
        (r"(?i)\bmis(?:remember|understood|taken)\b", 0.90, "misremember/misunderstood"),
        (r"(?i)\bwhen\s+did\s+I\s+(?:say|tell|mention)\b", 0.90, "when did I say"),
        (r"(?i)\bthat'?s?\s+not\s+what\s+I\s+(?:said|meant)\b", 0.85, "not what I said"),
        (r"(?i)\byou'?re\s+(?:confusing|mixing)\b", 0.80, "you're confusing"),
    ],
    # ── feedback_detector: useful memory patterns ──
    useful_feedback=[
        (r"(?i)\b(?:that'?s?\s+)?(?:helpful|useful)\b", 0.80, "helpful/useful"),
        (r"(?i)\bthanks?\s+for\s+(?:remembering|reminding)\b", 0.80, "thanks for remembering"),
        (r"(?i)\bgood\s+(?:to\s+)?(?:know|point)\b", 0.75, "good to know"),
        (r"(?i)\bthat\s+(?:helps|helped)\b", 0.75, "that helps"),
        (
            r"(?i)\b(?:oh\s+)?(?:right|yeah),?\s+(?:I\s+)?(?:forgot|remember)\b",
            0.75,
            "oh right I forgot",
        ),
    ],
    # ── feedback_detector: correction pattern names ──
    correction_names=frozenset(
        {
            "no that's not/wrong",
            "that's wrong/incorrect",
            "you're wrong",
            "misremember/misunderstood",
            "when did I say",
            "not what I said",
            "you're confusing",
            "I never said",
        }
    ),
    # ── policy / StateEncoder ──
    question_pattern=r"[?\uff1f]|\b(?:what|where|when|who|how|why|which|do you|can you|could you|is it|are you|have you)\b",  # noqa: E501
    personal_info_patterns=[
        r"(?i)\b(?:I'?m|I\s+am)\s+(.{2,30})",
        r"(?i)\bmy\s+(?:name|age|job|major|company|team|project)\b",
        r"(?i)\bI\s+(?:live|work|study)\s+(?:in|at)\b",
        r"(?i)\bI\s+(?:work|am\s+working)\s+(?:at|for|as)\b",
        r"(?i)\bI\s+(?:was\s+)?born\s+in\b",
    ],
    preference_patterns=[
        r"(?i)\bI\s+(?:like|love|enjoy|prefer)\b",
        r"(?i)\bI\s+(?:hate|dislike|don'?t\s+like)\b",
        r"(?i)\bmy\s+(?:favorite|favourite)\b",
        r"(?i)\bI'?m?\s+(?:into|fond\s+of|passionate\s+about)\b",
        r"(?i)\bI\s+(?:usually|always|often|tend\s+to)\b",
    ],
    tech_keywords=(
        r"(?<![a-zA-Z_])(?:"
        r"Python|Java(?:Script)?|TypeScript|Rust|Go|C\+\+|Ruby|Swift|Kotlin|Dart|"
        r"React|Vue|Angular|Next\.js|Django|Flask|FastAPI|Spring|Rails|"
        r"Docker|Kubernetes|k8s|AWS|GCP|Azure|Linux|Ubuntu|macOS|Windows|"
        r"MySQL|PostgreSQL|SQLite|Redis|MongoDB|Elasticsearch|"
        r"Git|GitHub|GitLab|CI/CD|DevOps|MLOps|"
        r"pandas|numpy|scipy|sklearn|scikit-learn|TensorFlow|PyTorch|Keras|"
        r"LLM|GPT|Claude|Gemini|machine\s+learning|deep\s+learning|AI|"
        r"API|REST|GraphQL|WebSocket|gRPC|"
        r"algorithm|data\s+structure|database|cloud|microservice"
        r")(?![a-zA-Z_])"
    ),
    emotion_keywords=(
        r"(?i)\b(?:happy|sad|angry|scared|anxious|excited|worried|"
        r"stressed|difficult|love|hate|enjoy|joyful|depressed|tired|thrilled)\b"
    ),
    # ── policy / MemoryPolicyAgent ──
    discourse_markers=[
        "by the way",
        "speaking of which",
        "anyway",
        "anyhow",
        "on another note",
        "that reminds me",
        "oh and",
        "also",
        "plus",
        "so",
        "well",
        "actually",
        "anyway though",
        "moving on",
        "changing the subject",
        "on a different note",
        "come to think of it",
        "that said",
        "but anyway",
    ],
    # ── resolution ──
    subject_markers=r"([\w]+(?:\s+(?:is|are|was|were|has|have|does|do)))",
    predicate_patterns=(r"([\w]+)\s+(?:is|are|was|were)\s+([\w\s]+)"),
    verb_endings=r"([\w]+(?:ing|ed|es|s|tion|ment|ness|ity|ful|less|able|ible))\b",
    verb_endings_fallback="",
    subject_strip=r"\s+(?:is|are|was|were|has|have|does|do)$",
    # ── graph_retriever ──
    word_extraction_pattern=r"\b\w{3,}\b",
    negative_predicates=["hate", "dislike", "can't", "don't", "refuse", "avoid"],
    # ── implicit_detector ──
    dismissive_pattern=r"(?i)^(?:ok(?:ay)?|sure|fine|yeah|yep|mhm|uh[\s-]?huh|alright|got\s+it|I\s+see|whatever)[.!]?$",
    # ── simple feedback lists ──
    sentence_endings=[],
    first_person=[
        "I",
        "I'm",
        "I've",
        "I'll",
        "I'd",
        "me",
        "my",
        "mine",
        "myself",
    ],
    emphasis=[
        "really",
        "very",
        "so",
        "extremely",
        "incredibly",
        "absolutely",
        "totally",
        "completely",
        "quite",
        "especially",
        "truly",
        "definitely",
    ],
    preference_expressions=["always", "every time", "whenever", "without fail"],
    constraint_expressions=[
        "never",
        "absolutely not",
        "no way",
        "don't ever",
        "under no circumstances",
        "please don't",
    ],
    request_patterns=[
        "please",
        "could you",
        "can you",
        "would you",
        "I'd like",
        "I want",
        "I need",
    ],
    positive_feedback_simple=[
        "right",
        "correct",
        "exactly",
        "perfect",
        "yes",
        "that's right",
        "good",
        "great",
        "thanks",
        "thank you",
        "you remember",
    ],
    negative_feedback_simple=[
        "no",
        "wrong",
        "incorrect",
        "that's not right",
        "I never said that",
        "that's not what I said",
        "you're confused",
        "you misunderstood",
    ],
    clarification_patterns=[
        "what do you mean",
        "I don't understand",
        "could you explain",
        "what does that mean",
        "I'm confused",
        "can you clarify",
        "for example",
        "say that again",
    ],
    elaboration_patterns=[
        "tell me more",
        "go on",
        "continue",
        "and then",
        "what else",
        "more details",
        "can you elaborate",
    ],
    fact_utterance_patterns=["is", "am", "are", "was", "were"],
    preference_utterance_patterns=[
        "like",
        "love",
        "enjoy",
        "prefer",
        "hate",
        "dislike",
        "want",
        "wish",
        "hope",
    ],
    experience_utterance_patterns=[
        "did",
        "have done",
        "tried",
        "experienced",
        "visited",
        "used",
        "worked with",
        "been to",
    ],
    chars_per_token=4.0,
    substantive_pos_tags=frozenset(),
    proper_noun_tag="",
    common_noun_tags=frozenset(),
)

register("en", _PATTERNS)
