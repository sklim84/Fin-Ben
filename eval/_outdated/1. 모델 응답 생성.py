import pandas as pd
import json
import requests
import os

OLLAMA_URL = "http://ollama:11434/api/generate"

# =================================
# 함수 1: 데이터 기반으로 question 생성
# =================================
def generate_answer(model, question, A, B, C, D, E):
    prompt = f"""
    다음은 객관식 질문과 그에 대한 선택지입니다.
    질문과 선택지를 주의깊게 읽고, 정답이라 생각하는 선택지의 기호(A,B,C,D,E)만 출력하세요.

    질문:
    {question}

    선택지:
    A. {A}
    B. {B}
    C. {C}
    D. {D}
    E. {E}

    반드시 정답의 기호(A,B,C,D,E)만 출력하세요.
    출력 형식은 단답형으로 작성하며, 반드시 아래와 같은 예시를 따르세요:
    A
    """

    data = {
        "model" : model,
        "prompt" : prompt,
        "stream" : False,
        "system" : "You are a helpful assistant.",
        "think": False,
        "stop": ["<think></think>"],
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
    for index, row in data.iterrows():
        try:
            _id = row['id']
            question = row['question']
            A = row['A']
            B = row['B']
            C = row['C']
            D = row['D']
            E = row['E']
            gold = row['gold']
            table = row['table']

            answer_70b = generate_answer("deepseek-r1:70b", question, A, B, C, D, E)
            answer_30b = generate_answer("qwen3:30b-a3b-instruct-2507-fp16", question, A, B, C, D, E)
            
            print(f"70B : {answer_70b}, 30B : {answer_30b}")
            
            # 결과 저장
            results.append({
                "id" : _id,
                "question" : question,
                "A" : A,
                "B" : B,
                "C" : C,
                "D" : D,
                "E" : E,
                "gold" : gold,
                "table" : table,
                "deepseek-r1:70b_answer" : answer_70b,
                "qwen3:30b-a3b-instruct-2507-fp16_answer" : answer_30b
            })
            
            num += 1
                
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
    file_path = "./data"
    file_list = os.listdir(file_path)

    for file_name in file_list:
      if file_name == '.ipynb_checkpoints':
        continue

      print(f"{file_name} Start")
      input_csv_path = f"{file_path}/{file_name}"   # 입력 CSV 파일 경로
      output_csv_path = f"./output/{file_name}_70b_30b_response.csv"  # 출력 CSV 파일 경로 (결과 저장용)

      # 데이터 생성 - CSV 처리 실행 -> 수행완료
      process_csv(input_csv_path, output_csv_path)


      print(f"{file_name} Done")
      print()