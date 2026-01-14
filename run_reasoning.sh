# 타겟 모델별 reasoning answer 생성
# opensource llm
# python3 -u eval/2_1_gen_reasoning_openlm.py
# proprietary (OpenAI) llm
# python3 -u eval/2_1_gen_reasoning_openai.py
# proprietary (Anthropic) llm
# python3 -u eval/2_1_gen_reasoning_claude.py

# 타겟 모델별 reasoning answer 평가
python3 -u eval/2_2_eval_reasoning_openai.py
