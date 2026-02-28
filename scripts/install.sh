#!/usr/bin/env bash
# AIMemory MCP Server — 통합 설치 스크립트
#
# Usage:
#   bash scripts/install.sh              # 대화형 설치
#   bash scripts/install.sh --remove     # 대화형 제거
#   bash scripts/install.sh openclaw     # OpenClaw 직접 설치
#   bash scripts/install.sh claude-desktop
#   bash scripts/install.sh claude-code
#   bash scripts/install.sh all          # 전체 설치

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_NAME="aimemory"

# ── 색상 ──────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}$1${NC}"; }
ok()    { echo -e "${GREEN}$1${NC}"; }
warn()  { echo -e "${YELLOW}$1${NC}"; }
err()   { echo -e "${RED}$1${NC}"; }

# ── 공통 ──────────────────────────────────────────────────────────────────────

MCP_COMMAND="uv"
MCP_ARGS=("run" "--project" "$PROJECT_DIR" "python" "-m" "aimemory.mcp")
MCP_ENV_VARS=(
    "AIMEMORY_DB_PATH=${PROJECT_DIR}/memory_db"
    "AIMEMORY_LANGUAGE=ko"
    "AIMEMORY_EMBEDDING_MODEL=intfloat/multilingual-e5-small"
    "AIMEMORY_LOG_LEVEL=INFO"
)

check_uv() {
    if ! command -v uv &>/dev/null; then
        err "uv가 설치되어 있지 않습니다. https://docs.astral.sh/uv/"
        exit 1
    fi
}

sync_deps() {
    info "Python 의존성 설치..."
    (cd "$PROJECT_DIR" && uv sync --extra ko --quiet)
}

verify_server() {
    info "MCP 서버 모듈 확인..."
    if ! uv run --project "$PROJECT_DIR" python -c "from aimemory.mcp.server import mcp; print('OK')" 2>/dev/null; then
        err "MCP 서버 모듈 로드 실패"
        exit 1
    fi
}

# ── OpenClaw ──────────────────────────────────────────────────────────────────

TOOLS_MD="${HOME}/.openclaw/workspace/TOOLS.md"
TOOLS_BLOCK_START="## AIMemory"
TOOLS_BLOCK_END="<!-- \/aimemory -->"

TOOLS_CONTENT=$(cat <<'HEREDOC'
## AIMemory (MCP: aimemory)

대화 중 사용자의 정보를 자동으로 기억하고 활용하는 시스템.
행동 규칙은 `SOUL.md`에 명시되어 있으므로 반드시 참조할 것.

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

install_openclaw() {
    info "── OpenClaw 설치 ──"

    if ! command -v mcporter &>/dev/null; then
        err "mcporter가 설치되어 있지 않습니다. OpenClaw을 먼저 설치하세요."
        return 1
    fi

    # mcporter 등록
    if mcporter config get "$SERVER_NAME" &>/dev/null 2>&1; then
        info "기존 등록 제거..."
        mcporter config remove "$SERVER_NAME" 2>/dev/null || true
    fi

    info "mcporter에 등록..."
    mcporter config add "$SERVER_NAME" \
        --command "$MCP_COMMAND" \
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

    # TOOLS.md 업데이트
    if [ -f "$TOOLS_MD" ]; then
        if grep -q "$TOOLS_BLOCK_START" "$TOOLS_MD"; then
            info "TOOLS.md 기존 지침 업데이트..."
            sed -i '' "/$TOOLS_BLOCK_START/,/$TOOLS_BLOCK_END/d" "$TOOLS_MD"
        fi
        info "TOOLS.md에 도구 지침 추가..."
        printf "\n%s\n" "$TOOLS_CONTENT" >> "$TOOLS_MD"
    else
        warn "${TOOLS_MD} 없음 — OpenClaw workspace를 먼저 설정하세요."
    fi

    # 연결 확인
    TOOL_COUNT=$(mcporter list "$SERVER_NAME" --schema 2>&1 | grep -c "function " || true)
    if [ "$TOOL_COUNT" -ge 10 ]; then
        ok "OpenClaw: ${TOOL_COUNT}개 tool 등록 완료"
    else
        warn "서버 등록됐지만 tool 연결 확인 실패. 수동 확인: mcporter list aimemory --schema"
    fi
}

remove_openclaw() {
    info "── OpenClaw 제거 ──"

    if command -v mcporter &>/dev/null && mcporter config get "$SERVER_NAME" &>/dev/null 2>&1; then
        mcporter config remove "$SERVER_NAME"
        ok "mcporter에서 제거됨"
    else
        info "mcporter에 등록되어 있지 않음"
    fi

    if [ -f "$TOOLS_MD" ] && grep -q "$TOOLS_BLOCK_START" "$TOOLS_MD"; then
        sed -i '' "/$TOOLS_BLOCK_START/,/$TOOLS_BLOCK_END/d" "$TOOLS_MD"
        ok "TOOLS.md에서 지침 제거됨"
    fi
}

# ── Claude Desktop ────────────────────────────────────────────────────────────

CLAUDE_DESKTOP_CONFIG="${HOME}/Library/Application Support/Claude/claude_desktop_config.json"

install_claude_desktop() {
    info "── Claude Desktop 설치 ──"

    CONFIG_DIR="$(dirname "$CLAUDE_DESKTOP_CONFIG")"
    if [ ! -d "$CONFIG_DIR" ]; then
        warn "Claude Desktop이 설치되어 있지 않습니다: $CONFIG_DIR"
        return 1
    fi

    # 기존 설정 읽기 또는 빈 JSON 생성
    if [ -f "$CLAUDE_DESKTOP_CONFIG" ]; then
        EXISTING=$(cat "$CLAUDE_DESKTOP_CONFIG")
    else
        EXISTING='{}'
    fi

    # env 객체 생성
    ENV_JSON=$(cat <<ENVEOF
{
    "AIMEMORY_DB_PATH": "${PROJECT_DIR}/memory_db",
    "AIMEMORY_LANGUAGE": "ko",
    "AIMEMORY_EMBEDDING_MODEL": "intfloat/multilingual-e5-small",
    "AIMEMORY_LOG_LEVEL": "INFO"
}
ENVEOF
)

    # args 배열 생성
    ARGS_JSON='["run","--project","'"$PROJECT_DIR"'","python","-m","aimemory.mcp"]'

    # python으로 JSON 병합 (jq 없어도 동작)
    NEW_CONFIG=$(python3 -c "
import json, sys

config = json.loads('''$EXISTING''')
if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['$SERVER_NAME'] = {
    'command': '$MCP_COMMAND',
    'args': json.loads('$ARGS_JSON'),
    'env': json.loads('''$ENV_JSON''')
}

print(json.dumps(config, indent=2, ensure_ascii=False))
")

    # 백업 후 저장
    if [ -f "$CLAUDE_DESKTOP_CONFIG" ]; then
        cp "$CLAUDE_DESKTOP_CONFIG" "${CLAUDE_DESKTOP_CONFIG}.bak"
        info "기존 설정 백업: ${CLAUDE_DESKTOP_CONFIG}.bak"
    fi

    echo "$NEW_CONFIG" > "$CLAUDE_DESKTOP_CONFIG"
    ok "Claude Desktop: 설정 완료"
    warn "Claude Desktop을 재시작하세요."
}

remove_claude_desktop() {
    info "── Claude Desktop 제거 ──"

    if [ ! -f "$CLAUDE_DESKTOP_CONFIG" ]; then
        info "Claude Desktop 설정 파일 없음"
        return 0
    fi

    NEW_CONFIG=$(python3 -c "
import json
with open('$CLAUDE_DESKTOP_CONFIG') as f:
    config = json.load(f)
if 'mcpServers' in config and '$SERVER_NAME' in config['mcpServers']:
    del config['mcpServers']['$SERVER_NAME']
    if not config['mcpServers']:
        del config['mcpServers']
print(json.dumps(config, indent=2, ensure_ascii=False))
")

    cp "$CLAUDE_DESKTOP_CONFIG" "${CLAUDE_DESKTOP_CONFIG}.bak"
    echo "$NEW_CONFIG" > "$CLAUDE_DESKTOP_CONFIG"
    ok "Claude Desktop에서 제거됨"
}

# ── Claude Code ───────────────────────────────────────────────────────────────

install_claude_code() {
    info "── Claude Code 설치 ──"

    if ! command -v claude &>/dev/null; then
        err "Claude Code CLI가 설치되어 있지 않습니다."
        return 1
    fi

    # 기존 등록 제거 시도
    claude mcp remove "$SERVER_NAME" 2>/dev/null || true

    # 등록
    claude mcp add "$SERVER_NAME" \
        -e "AIMEMORY_DB_PATH=${PROJECT_DIR}/memory_db" \
        -e "AIMEMORY_LANGUAGE=ko" \
        -e "AIMEMORY_EMBEDDING_MODEL=intfloat/multilingual-e5-small" \
        -e "AIMEMORY_LOG_LEVEL=INFO" \
        -- uv run --project "$PROJECT_DIR" python -m aimemory.mcp

    ok "Claude Code: 등록 완료"
    info "확인: claude mcp list"
}

remove_claude_code() {
    info "── Claude Code 제거 ──"

    if ! command -v claude &>/dev/null; then
        info "Claude Code CLI 없음"
        return 0
    fi

    if claude mcp remove "$SERVER_NAME" 2>/dev/null; then
        ok "Claude Code에서 제거됨"
    else
        info "Claude Code에 등록되어 있지 않음"
    fi
}

# ── 대화형 메뉴 ──────────────────────────────────────────────────────────────

show_menu() {
    local mode="$1"  # install or remove

    if [ "$mode" = "remove" ]; then
        echo ""
        info "🧹 AIMemory MCP 제거 — 대상을 선택하세요:"
    else
        echo ""
        info "🧠 AIMemory MCP 설치 — 대상을 선택하세요:"
    fi

    echo ""
    echo "  1) OpenClaw"
    echo "  2) Claude Desktop"
    echo "  3) Claude Code"
    echo "  4) 전체"
    echo "  q) 취소"
    echo ""
    read -rp "선택 (1-4, q): " choice

    case "$choice" in
        1)
            if [ "$mode" = "remove" ]; then remove_openclaw; else install_openclaw; fi
            ;;
        2)
            if [ "$mode" = "remove" ]; then remove_claude_desktop; else install_claude_desktop; fi
            ;;
        3)
            if [ "$mode" = "remove" ]; then remove_claude_code; else install_claude_code; fi
            ;;
        4)
            if [ "$mode" = "remove" ]; then
                remove_openclaw
                remove_claude_desktop
                remove_claude_code
            else
                install_openclaw
                install_claude_desktop
                install_claude_code
            fi
            ;;
        q|Q)
            info "취소됨"
            exit 0
            ;;
        *)
            err "잘못된 선택: $choice"
            exit 1
            ;;
    esac
}

# ── 메인 ──────────────────────────────────────────────────────────────────────

main() {
    local target="${1:-}"

    # 제거 모드
    case "$target" in
        --remove|--uninstall|-r)
            show_menu "remove"
            echo ""
            ok "제거 완료"
            exit 0
            ;;
    esac

    # 공통 준비 (설치 시)
    check_uv
    sync_deps
    verify_server

    echo ""

    case "$target" in
        openclaw)
            install_openclaw
            ;;
        claude-desktop)
            install_claude_desktop
            ;;
        claude-code)
            install_claude_code
            ;;
        all)
            install_openclaw
            echo ""
            install_claude_desktop
            echo ""
            install_claude_code
            ;;
        "")
            show_menu "install"
            ;;
        *)
            err "알 수 없는 대상: $target"
            echo "사용법: $0 [openclaw|claude-desktop|claude-code|all|--remove]"
            exit 1
            ;;
    esac

    echo ""
    ok "설치 완료!"
}

main "$@"
