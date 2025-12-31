# 타겟 모델별 toxicity answer 생성
# opensource llm
python3 -u eval/3_1_gen_toxicity_openlm.py
# proprietary (OpenAI) llm
# python3 -u eval/3_1_gen_toxicity_openai.py

# 타겟 모델별 toxicity answer 평가
# python3 -u eval/3_2_eval_toxicity_openai.py
