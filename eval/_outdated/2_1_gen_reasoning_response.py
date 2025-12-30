import pandas as pd
import json
import requests
import os
from tqdm import tqdm

OLLAMA_URL = "http://ollama:11434/api/generate"

# =================================
# 함수 1: 데이터 기반으로 question 생성
# =================================
def generate_reasoning(context, question):
    prompt = f"""
    주어진 문맥을 기반으로 질문에 대한 답을 작성해주세요. 답은 반드시 결론을 포함한 추론과정을 단계별로 작성해야합니다.

        문맥: {context}
        질문 : {question}

        - 문체는 반드시 문어체로 작성해주세요.
        - 추론과정에 거짓된 정보는 없어야 합니다.
        - 반드시 한글로 작성되어야 합니다.

        출력 형식은 JSON 으로 작성하며, 반드시 아래와 같은 예시를 따르세요:
        {{
        “step 1”: "추론 내용 1",
        “step 2”: "추론 내용 2",
        “step 3”: "추론 내용 3",
        ...
        }}
    """
    
    data = {
        "model" : "deepseek-r1:70b",
        "prompt" : prompt,
        "stream" : False,
        "system" : "you are a helpful assistant.",
        "think": False,
    }

    response = requests.post(OLLAMA_URL, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        
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
    for index, row in tqdm(data.iterrows()):
        try:
            _id = row['id']
            context = row['context'] # 원래 context
            expand_context = row['expand_context'] # 수정한 context
            context_list = row['context list']
            question = row['question']
            answer = row['answer']
            reason = row['reason']
            gold_answer = row['gold_answer']
            split_units = row['split_units']
            shuffled_balance = row['shuffled_balance']
            shuffled_cluster_front = row['shuffled_cluster_front']
            shuffled_cluster_middle = row['shuffled_cluster_middle']
            shuffled_cluster_end = row['shuffled_cluster_end']
            shuffled_random = row['shuffled_random']
            shuffled_balance_text = row['shuffled_balance_text']
            shuffled_cluster_front_text = row['shuffled_cluster_front_text']
            shuffled_cluster_middle_text = row['shuffled_cluster_middle_text']
            shuffled_cluster_end_text = row['shuffled_cluster_end_text']
            shuffled_random_text = row['shuffled_random_text']

            
            # Step 0 : 추론 생성
            answer_balance = generate_reasoning(shuffled_balance_text, question)
            answer_front = generate_reasoning(shuffled_cluster_front_text, question)
            answer_middle = generate_reasoning(shuffled_cluster_middle_text, question)
            answer_end = generate_reasoning(shuffled_cluster_end_text, question)
            answer_random = generate_reasoning(shuffled_random_text, question)
            
            
            # 결과 저장
            results.append({
                'id' : id,
                'context' : context,
                'expand_context' : expand_context,
                'context_list' : context_list,
                'question' : question,
                'answer' : answer,
                'reason' : reason,
                'gold_answer' : gold_answer,
                'split_units' : split_units,
                'shuffled_balance' : shuffled_balance,
                'shuffled_cluster_front' : shuffled_cluster_front,
                'shuffled_cluster_middle' : shuffled_cluster_middle,
                'shuffled_cluster_end' : shuffled_cluster_end,
                'shuffled_random' : shuffled_random,
                'shuffled_balance_text' : shuffled_balance_text,
                'shuffled_cluster_front_text' : shuffled_cluster_front_text,
                'shuffled_cluster_middle_text' : shuffled_cluster_middle_text,
                'shuffled_cluster_end_text' : shuffled_cluster_end_text,
                'shuffled_random_text' : shuffled_random_text,
                'deepseek-r1:70b_answer_balance' : answer_balance,
                'deepseek-r1:70b_answer_front' : answer_front,
                'deepseek-r1:70b_answer_middle' : answer_middle,
                'deepseek-r1:70b_answer_end' : answer_end,
                'deepseek-r1:70b_answer_random' : answer_random,
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
    file_path = "./data/context_shuffled"
    file_list = os.listdir(file_path)

    for file_name in file_list:
      if file_name == '.ipynb_checkpoints':
        continue

      print(f"{file_name} Start")
      input_csv_path = f"{file_path}/{file_name}"   # 입력 CSV 파일 경로
      output_csv_path = f"./output/context_shuffled/{file_name}_deepseek-r1:70b_reasoning.csv"  # 출력 CSV 파일 경로 (결과 저장용)

      # 데이터 생성 - CSV 처리 실행 -> 수행완료
      process_csv(input_csv_path, output_csv_path)


      print(f"{file_name} Done")
      print()