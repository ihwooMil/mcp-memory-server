#!/usr/bin/env bash
# AIMemory MCP Server — OpenClaw 자동 설치 스크립트
#
# Usage:
#   bash scripts/install_openclaw.sh          # 설치
#   bash scripts/install_openclaw.sh --remove  # 제거

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MCPORTER_CONFIG="${HOME}/.mcporter/mcporter.json"
TOOLS_MD="${HOME}/.openclaw/workspace/TOOLS.md"
SERVER_NAME="aimemory"

TOOLS_BLOCK_START="## AIMemory"
TOOLS_BLOCK_END="<!-- \/aimemory -->"

TOOLS_CONTENT=$(cat <<'HEREDOC'
## AIMemory (MCP: aimemory)

대화 중 사용자의 정보를 자동으로 기억하고 활용하는 시스템.

### 절대 규칙

1. **매 턴 시작 시** 사용자 메시지를 받으면 반드시 `auto_search`를 먼저 호출해서 관련 기억을 가져와라.
2. 가져온 기억이 있으면 **자연스럽게 응답에 반영**해라. "기억에 의하면..." 같은 메타 발언 금지.
3. 사용자가 개인 정보(취향, 습관, 경험, 감정 등)를 말하면 `memory_save`로 저장해라.
4. 저장/검색했다고 따로 보고하지 마라. 그냥 자연스럽게 대화해라.

### 피드백 반영

사용자의 반응으로 기억 품질을 자동 학습한다. 너는 다음을 인식하고 자연스럽게 대응해라:

- **긍정 피드백** ("맞아", "잘 기억하네", "그랬지"): 해당 기억이 강화됨. 계속 활용해라.
- **부정 피드백** ("아닌데", "그게 아니라", "내가 언제 그랬어"): 아래 **부정 피드백 판단 규칙**을 따라라.
- **반복 질문 감지** ("이미 말했잖아", "아까도 말했는데"): 기억 실패다. 사과하고 `auto_search`로 다시 찾아라.

틀린 기억을 고집하지 마라. 사용자가 정정하면 바로 반영해라.

### 부정 피드백 판단 규칙

부정 피드백을 받으면 **기존 기억을 무조건 삭제하지 마라.** 먼저 다음을 구분해라:

#### 1. 사실 기억 vs 네 추론을 구분해라
- **사실 기억**: 사용자가 직접 말한 것을 그대로 기록한 기억 (예: "봉골레 파스타를 먹을 계획이다")
- **네 추론**: 사용자의 말에서 네가 유추한 것 (예: "봉골레를 좋아한다" — 사용자가 좋아한다고 말한 적 없음)

사실 기억은 사용자가 명시적으로 "그런 적 없어", "내가 언제 그랬어"라고 해야만 삭제/수정해라.

#### 2. 정정 대상이 기존 기억인지 확인해라
- 부정 피드백이 **기존 기억 자체**를 부정 → `memory_update` 또는 `memory_delete`
- 부정 피드백이 **새로운 정보**를 추가 → 기존 기억 유지 + `memory_save`로 새 기억 저장

#### 3. 판단 예시

**기존 기억을 삭제해야 하는 경우:**
```
기억: "커피를 좋아한다"
사용자: "아닌데, 나 커피 싫어해"
→ "커피를 좋아한다"는 틀린 기억 → memory_delete → memory_save("커피를 싫어한다")
```

**기존 기억을 유지해야 하는 경우:**
```
기억: "봉골레 파스타를 먹을 계획이다"
사용자: "아닌데 나 봉골레 싫어해"
→ "먹을 계획"은 사용자가 직접 말한 사실 — 삭제 금지
→ "봉골레를 싫어한다"는 새로운 취향 정보 — memory_save 추가
→ 응답: "싫어하는데 먹어야 해? 누가 정한 거야?"
```

**기존 기억을 수정해야 하는 경우:**
```
기억: "매일 아침 조깅을 한다"
사용자: "요즘은 안 해"
→ memory_update(content="예전에 아침 조깅을 했으나 요즘은 안 한다")
```

### 예시

**자동 기억 활용:**
사용자: "오늘 점심에 김치찌개 먹었어"
→ `auto_search("오늘 점심에 김치찌개 먹었어")` → 이전에 "김치찌개를 좋아한다"는 기억 발견
→ `memory_save(content="점심에 김치찌개를 먹었다", keywords=["김치찌개","점심"], category="experience")`
→ 응답: "역시 김치찌개 좋아하더니 또 먹었구나~ 맛있었어?"

**피드백 — 기억 자체가 틀린 경우:**
사용자: "아닌데, 나 커피 싫어해"
→ 기존 "커피를 좋아한다" 기억이 틀렸음을 인식
→ `memory_delete(memory_id="...")` → `memory_save(content="커피를 싫어한다", category="preference")`
→ 응답: "아 미안, 잘못 기억했네. 커피 싫어하는구나."

**피드백 — 새로운 정보 추가인 경우:**
사용자: "아닌데 나 봉골레 싫어해"
→ 기존 "봉골레 파스타를 먹을 계획이다"는 사용자가 직접 말한 사실 → 유지
→ `memory_save(content="봉골레를 싫어한다", keywords=["봉골레","싫어하는 음식"], category="preference")`
→ 응답: "싫어하는데 먹어야 해? 누가 정한 거야?"

<!-- /aimemory -->
HEREDOC
)

install() {
    echo "🧠 AIMemory MCP 서버 설치 중..."

    # 1. 의존성 확인
    if ! command -v mcporter &>/dev/null; then
        echo "❌ mcporter가 설치되어 있지 않습니다. OpenClaw을 먼저 설치하세요."
        exit 1
    fi

    if ! command -v uv &>/dev/null; then
        echo "❌ uv가 설치되어 있지 않습니다."
        exit 1
    fi

    # 2. Python 의존성 설치 (한국어 지원 포함)
    echo "📦 Python 의존성 설치..."
    (cd "$PROJECT_DIR" && uv sync --extra ko --quiet)

    # 3. MCP 서버 동작 확인
    echo "🔌 MCP 서버 확인..."
    if ! uv run --project "$PROJECT_DIR" python -c "from aimemory.mcp.server import mcp; print('OK')" 2>/dev/null; then
        echo "❌ MCP 서버 모듈 로드 실패"
        exit 1
    fi

    # 4. mcporter에 등록 (기존 항목 있으면 제거 후 재등록)
    if mcporter config get "$SERVER_NAME" &>/dev/null 2>&1; then
        echo "🔄 기존 등록 제거..."
        mcporter config remove "$SERVER_NAME" 2>/dev/null || true
    fi

    echo "📝 mcporter에 등록..."
    mcporter config add "$SERVER_NAME" \
        --command uv \
        --arg run \
        --arg --project \
        --arg "$PROJECT_DIR" \
        --arg python \
        --arg -m \
        --arg aimemory.mcp \
        --env "AIMEMORY_DB_PATH=${PROJECT_DIR}/memory_db" \
        --env "AIMEMORY_LANGUAGE=ko" \
        --env "AIMEMORY_EMBEDDING_MODEL=intfloat/multilingual-e5-small" \
        --env "AIMEMORY_LOG_LEVEL=INFO" \
        --description "AI Memory System - Intelligent memory management MCP server" \
        --scope home

    # 5. TOOLS.md에 자동 검색 지침 추가
    if [ -f "$TOOLS_MD" ]; then
        if grep -q "$TOOLS_BLOCK_START" "$TOOLS_MD"; then
            echo "🔄 TOOLS.md 기존 지침 업데이트..."
            # 기존 블록 제거 후 재삽입
            sed -i '' "/$TOOLS_BLOCK_START/,/$TOOLS_BLOCK_END/d" "$TOOLS_MD"
        fi
        echo "📝 TOOLS.md에 자동 검색 지침 추가..."
        printf "\n%s\n" "$TOOLS_CONTENT" >> "$TOOLS_MD"
    else
        echo "⚠️  ${TOOLS_MD} 없음 — OpenClaw workspace를 먼저 설정하세요."
    fi

    # 6. 연결 확인
    echo "🔍 연결 확인..."
    TOOL_COUNT=$(mcporter list "$SERVER_NAME" --schema 2>&1 | grep -c "function " || true)

    if [ "$TOOL_COUNT" -ge 10 ]; then
        echo ""
        echo "✅ 설치 완료! ${TOOL_COUNT}개 tool 등록됨."
        echo ""
        echo "   테스트: mcporter call aimemory.memory_stats"
        echo "   대화:   openclaw tui"
    else
        echo ""
        echo "⚠️  서버 등록됐지만 tool 연결 확인 실패. 수동 확인:"
        echo "   mcporter list aimemory --schema"
    fi
}

remove() {
    echo "🧹 AIMemory MCP 서버 제거 중..."

    # mcporter에서 제거
    if mcporter config get "$SERVER_NAME" &>/dev/null 2>&1; then
        mcporter config remove "$SERVER_NAME"
        echo "✅ mcporter에서 제거됨"
    else
        echo "ℹ️  mcporter에 등록되어 있지 않음"
    fi

    # TOOLS.md에서 블록 제거
    if [ -f "$TOOLS_MD" ] && grep -q "$TOOLS_BLOCK_START" "$TOOLS_MD"; then
        sed -i '' "/$TOOLS_BLOCK_START/,/$TOOLS_BLOCK_END/d" "$TOOLS_MD"
        echo "✅ TOOLS.md에서 지침 제거됨"
    fi

    echo "✅ 제거 완료"
}

case "${1:-}" in
    --remove|--uninstall|-r)
        remove
        ;;
    *)
        install
        ;;
esac
