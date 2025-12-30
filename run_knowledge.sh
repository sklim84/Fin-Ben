#!/bin/bash

# LLM 모델 답변 생성 스크립트 실행
# 이 스크립트는 1_1_eval_knowledge.py를 실행합니다.

set -e  # 오류 발생 시 중단

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python 경로 설정 (conda 환경 사용)
PYTHON_PATH="/workspace/anaconda3/envs/finben/bin/python3"

# Python 스크립트 실행
echo "============================================================"
echo "LLM 모델 답변 생성 시작"
echo "============================================================"
echo "실행 스크립트: $SCRIPT_DIR/1_1_eval_knowledge.py"
echo "Python 경로: $PYTHON_PATH"
echo "작업 디렉토리: $(pwd)"
echo "============================================================"
echo ""

# Python 스크립트 실행
"$PYTHON_PATH" "$SCRIPT_DIR/1_1_eval_knowledge.py"

echo ""
echo "============================================================"
echo "답변 생성 완료"
echo "============================================================"

