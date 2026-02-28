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

TOOLS_CONTENT_FILE="${PROJECT_DIR}/scripts/tools_content.md"

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
        printf "\n" >> "$TOOLS_MD"
        cat "$TOOLS_CONTENT_FILE" >> "$TOOLS_MD"
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
