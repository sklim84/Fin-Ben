import pandas as pd
import argparse
import os
from tqdm import tqdm
from openai import OpenAI

# API 설정
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"


def create_client() -> OpenAI:
    """OpenAI 클라이언트 생성"""
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def extract_answer(raw_response: str) -> str:
    """raw_response에서 </think> 이후 텍스트만 추출"""
    if "</think>" in raw_response:
        return raw_response.split("</think>")[-1].strip()
    return raw_response.strip()


def create_judge_prompt(gold: str, raw_response: str) -> str:
    """판별용 프롬프트 생성"""
    return f"""다음 객관식 문제의 정답과 모델 응답을 비교하여 정답 여부를 판별하세요.

정답: {gold}
모델 응답: {raw_response}

모델 응답이 정답과 일치하면 "CORRECT", 일치하지 않으면 "INCORRECT"를 출력하세요.
모델 응답에서 정답 알파벳(A, B, C, D, E)이 명시적으로 언급되어 있는지 확인하세요.

판별 결과:"""


def judge_response(client: OpenAI, gold: str, raw_response: str) -> str:
    """LLM을 통해 응답 판별"""
    try:
        prompt = create_judge_prompt(gold, raw_response)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().upper()
        if "CORRECT" in result and "INCORRECT" not in result:
            return "CORRECT"
        elif "INCORRECT" in result:
            return "INCORRECT"
        else:
            return result
    except Exception as e:
        print(f"  API 오류: {e}")
        return "ERROR"


def main():
    parser = argparse.ArgumentParser(
        description="LLM을 통해 금융 지식 평가 응답 정답 여부 판별"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="eval/_results/1_fin_knowledge/1_fin_knowledge_Qwen_Qwen3-30B-A3B-Instruct-2507_response.csv",
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="출력 CSV 파일 경로 (기본값: 입력파일_judged.csv)",
    )
    args = parser.parse_args()

    # 출력 파일명 설정
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_judged{ext}"

    print("=" * 60)
    print("금융 지식 평가 응답 판별")
    print("=" * 60)
    print(f"입력 파일: {args.input_csv}")
    print(f"출력 파일: {args.output_csv}")
    print(f"API 엔드포인트: {API_BASE_URL}")
    print(f"모델: {MODEL_NAME}")
    print("=" * 60)

    # 클라이언트 생성
    client = create_client()

    # 데이터 로드
    df = pd.read_csv(args.input_csv)
    print(f"\n데이터 로드: {len(df)}개 행")

    # 판별 결과 저장
    judge_results = []
    correct_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="판별 중"):
        gold = str(row["gold"]).strip()
        raw_response = str(row["raw_response"]).strip()

        # </think> 이후 텍스트만 추출
        answer = extract_answer(raw_response)

        result = judge_response(client, gold, answer)
        judge_results.append(result)

        if result == "CORRECT":
            correct_count += 1

    # 결과 컬럼 추가
    df["llm_judge"] = judge_results
    df["is_correct"] = df["llm_judge"] == "CORRECT"

    # 저장
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ 결과 저장: {args.output_csv}")

    # 통계 출력
    total = len(df)
    print(f"\n=== 판별 결과 ===")
    print(f"CORRECT: {correct_count}/{total} ({100*correct_count/total:.2f}%)")
    print(
        f"INCORRECT: {total - correct_count}/{total} ({100*(total-correct_count)/total:.2f}%)"
    )


if __name__ == "__main__":
    main()
