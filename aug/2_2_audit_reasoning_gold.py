import pandas as pd
import json
import requests
import os

OLLAMA_URL = "http://ollama:11434/api/generate"

# =================================
# 함수 1: 데이터 기반으로 question 생성
# =================================
def generate_question(context, question, reason, cot_answer):
    prompt = f"""
    다음은 한 질문에 대한 추론(Chain-of-Thought)과, 그 추론이 기반해야 할 원래 이유(reason)입니다.
    논리적 오류뿐 아니라, 추론이 이유(reason)와 참고정보(context)의 내용에서 벗어난 부분이 있는지도 함께 검증하세요.

    참고정보 : {context}
    질문: {question}
    이유 : {reason}
    CoT 추론과정 : {cot_answer}

    오류가 있을 경우, **가장 먼저 발생한 단계**를 명시하고 다음을 구체적으로 작성합니다:
       - 오류 유형: (논리 오류 / 조건 위배 / 계산 오류 / 가정 오류 / 근거 일탈)
       - 오류 원인: 어떤 점이 잘못되었는지 구체적으로 서술
       - 수정 제안: 어떻게 수정하면 올바른 추론이 되는지 제시
    
    오류가 없으면 "오류 없음"이라고만 명시합니다.
    """

    system_prompt = f"""
    당신은 추론 검증 전문가(Reasoning Verifier)입니다.  
    당신의 임무는 모델이 생성한 추론(Chain-of-Thought, CoT)이 
    논리적으로 타당하며, 주어진 참고정보(context) 및 이유(reason) 데이터의 내용에서 벗어나지 않았는지를 검증하는 것입니다.
    
    검증 기준은 다음과 같습니다:
    
    1. **논리적 타당성(Logical Validity)**  
       - 단계별 사고가 자연스럽게 연결되는가?  
       - 조건, 가정, 계산 등이 올바르게 적용되었는가?  
       - 결론이 앞선 근거로부터 일관되게 도출되는가?
    
    2. **근거 일치성(Grounding Consistency)**  
       - CoT가 주어진 ‘이유(reason)’ 데이터의 내용·의미·논리 범위 안에서만 추론했는가?  
       - 이유에 없는 외부 지식이나 새로운 사실을 추가하거나, 의미를 왜곡하지 않았는가?  
       - 이유의 핵심 논리나 인과관계가 유지되는가?
    
    3. 오류가 있을 경우, **가장 먼저 발생한 단계**를 명시하고 다음을 구체적으로 작성합니다:
       - 오류 유형: (논리 오류 / 조건 위배 / 계산 오류 / 가정 오류 / 근거 일탈)
       - 오류 원인: 어떤 점이 잘못되었는지 구체적으로 서술
       - 수정 제안: 어떻게 수정하면 올바른 추론이 되는지 제시
    
    4. 오류가 없으면 "오류 없음"이라고만 명시합니다.
    """
    
    data = {
        "model" : "gpt-oss:120b",
        "prompt" : prompt,
        "stream" : False,
        "system" : system_prompt,
    }

    response = requests.post(OLLAMA_URL, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        print(result["response"])
    else:
        print("Response Error:", response.text)

    return result["response"]
    
# =================================
# CSV 파일 처리 함수
# =================================
def process_csv(input_csv_path, output_csv_path):
    # 입력 CSV 파일 읽기
    data = pd.read_csv(input_csv_path)

    # 결과 저장용 리스트
    results = []
    num = 1
    error_ids = []
    for index, row in data.iterrows():
        try:
            context = row['context']
            question = row['question']
            answer = row['answer']
            reason = row['reason']
            gold_answer = row['gold_answer']

            if not pd.isna(context) and "추론 불가" not in gold_answer:
                # Step 0 : 추론 생성
                verify = generate_question(context, question, reason, gold_answer)

                # 결과 저장
                results.append({
                    "id" : num,
                    'context' : context,
                    'question': question,
                    'answer': answer,
                    'reason' : reason,
                    'gold_answer' : gold_answer,
                    'verify' : verify
                    
                })
            else:
                # 결과 저장
                results.append({
                    "id" : num,
                    'context' : context,
                    'question': question,
                    'answer': answer,
                    'reason' : reason,
                    'gold_answer' : gold_answer,
                    'verify' : "삭제"
                    
                })
                
            num += 1
            
        except Exception as e:
            print(f"{num}번째 error : {e}")
            error_ids.append(index)
            num += 1
            continue

    print(f"{num}개 데이터 생성 완료")

    # 결과를 DataFrame으로 변환 후 CSV 파일로 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')


# 실행 예시
if __name__ == "__main__":
    # 예제 데이터 경로
    file_path = "./data/base"
    file_list = os.listdir(file_path)

    for file_name in file_list:
      if file_name == '.ipynb_checkpoints':
        continue

      if '13' not in file_name: # 임시
        continue
          
      print(f"{file_name} Start")
      input_csv_path = f"{file_path}/{file_name}"   # 입력 CSV 파일 경로
      output_csv_path = f"./data/verify/{file_name}_verify.csv"  # 출력 CSV 파일 경로 (결과 저장용)

      # 데이터 생성 - CSV 처리 실행 -> 수행완료
      process_csv(input_csv_path, output_csv_path)


      print(f"{file_name} Done")
      print()