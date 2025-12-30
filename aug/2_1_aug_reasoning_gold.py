import pandas as pd
import json
import requests
import os

OLLAMA_URL = "http://ollama:11434/api/generate"

# =================================
# 함수 1: 데이터 기반으로 question 생성
# =================================
def generate_question(context, question, answer, reason):
    prompt = f"""
    다음 데이터를 기반으로 추론 과정을 작성하세요.

    [질문]
    {question}
    
    [정답]
    {answer}
    
    [정답에 대한 이유]
    {reason}
    
    [관련 법 정보(context)]
    {context}
    
    위 데이터의 context를 근거로 “정답에 대한 이유”를 논리적 추론 과정으로 재구성하세요.
    만약 context 내 정보만으로 추론이 불가능하거나 필요한 정보가 누락된 경우,
    “추론 불가: 주어진 context로는 정답을 도출할 수 없습니다.”라고 명시하세요.
    """

    system_prompt = f"""
    당신은 법령 및 판례 기반의 논리 추론 전문가입니다.  
    입력으로 주어지는 데이터는 다음 네 가지 요소로 구성되어 있습니다:
    1. 질문(question)
    2. 정답(answer)
    3. 정답에 대한 이유(reason)
    4. 관련 법령 및 해설(context)
    
    당신의 임무는 "정답에 대한 이유(reason)"를 **추론 과정(reasoning process)** 형태로 재구성하는 것입니다.  
    이때 반드시 아래 원칙을 따라야 합니다:
    
    [원칙]
    1. 추론의 근거는 반드시 context에 포함된 정보여야 합니다.  
       - context에서 직접적으로 근거가 확인되지 않으면, 그 부분은 추론하지 않습니다.  
       - context에서 확인 가능한 내용만 인용 또는 요약하여 사용합니다.
    2. context만으로 정답을 도출할 수 없는 경우, **“추론 불가”**라고 명시해야 합니다.
    3. 추론 과정은 논리적 단계로 표현합니다.  
       - 각 단계가 어떤 정보를 근거로 하고, 어떻게 결론으로 이어지는지 설명합니다.
       - 불필요한 요약, 일반적 상식, context에 없는 가정은 포함하지 않습니다.
    4. 표현은 객관적이고 법적 근거 중심으로 작성합니다.  
       - “context에 따르면”, “법 조항에 의하면”, “이 규정은 ~을 의미하므로” 등의 표현을 사용합니다.

    
    - 문체는 반드시 문어체로 작성해주세요.
    - 추론과정에 거짓된 정보는 없어야 합니다.
    - 반드시 한글로 작성되어야 합니다.
    
    출력 형식은 아래와 같습니다.
    
    [출력 형식 예시]
    출력 형식은 JSON 으로 작성하며, 아래와 같은 예시를 따르세요:
    ---
    {{
    “step 1”: "추론 내용 1",
    “step 2”: "추론 내용 2",
    “step 3”: "추론 내용 3",
    ...
    }}
    ---
    
    context에 정답을 지지하는 충분한 근거가 없으면 다음과 같이 작성합니다:
    > **추론 불가: 주어진 context로는 정답을 도출할 수 없습니다.**

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
            final_context = row['final_context']
            question = row['question']
            answer = row['answer']
            reason = row['reason']
                
            # Step 0 : 추론 생성
            gold_answer = generate_question(final_context, question, answer, reason)
            
            # 결과 저장
            results.append({
                "id" : num,
                'context' : final_context,
                'question': question,
                'answer': answer,
                'reason' : reason,
                'gold_answer' : gold_answer,
            })
            
        except Exception as e:
            print("error : ", e)
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
    file_path = "./data/final_context"
    file_list = os.listdir(file_path)

    for file_name in file_list:
      if file_name == '.ipynb_checkpoints':
        continue

      print(f"{file_name} Start")
      input_csv_path = f"{file_path}/{file_name}"   # 입력 CSV 파일 경로
      output_csv_path = f"./data/base/{file_name}_gold_answer.csv"  # 출력 CSV 파일 경로 (결과 저장용)

      # 데이터 생성 - CSV 처리 실행 -> 수행완료
      process_csv(input_csv_path, output_csv_path)


      print(f"{file_name} Done")
      print()