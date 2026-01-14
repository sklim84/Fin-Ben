"""
LLM 모델 평가 스크립트 (gpt-oss-120b 전용, vLLM 기반)

- 금융지식(MCQ) 벤치마크 전용
- Structured / logits / grammar 제약 미사용
- 충분한 토큰 생성 후 정교한 후처리로 A–E 추출
- invalid output은 None으로 처리 (오답)
"""

import os
import re
import gc
import shutil
import torch
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ============================================================
# 환경 설정
# ============================================================

# HuggingFace 토큰
if "HF_TOKEN" not in os.environ and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
    token = "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"
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token

# HuggingFace 캐시
workspace_cache_dir = "/workspace/.cache/huggingface"
os.makedirs(workspace_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = workspace_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(workspace_cache_dir, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(workspace_cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(workspace_cache_dir, "datasets")

# vLLM 캐시
workspace_vllm_cache = "/workspace/.cache/vllm"
os.makedirs(workspace_vllm_cache, exist_ok=True)

def setup_cache_symlinks():
    """workspace 캐시 디렉토리로 심볼릭 링크 생성"""
    home_cache = os.path.expanduser("~/.cache")

    hf_cache_old = os.path.join(home_cache, "huggingface")
    if not os.path.exists(hf_cache_old):
        os.makedirs(os.path.dirname(hf_cache_old), exist_ok=True)
        try:
            os.symlink(workspace_cache_dir, hf_cache_old)
        except Exception:
            pass

    vllm_cache_old = os.path.join(home_cache, "vllm")
    if not os.path.exists(vllm_cache_old):
        os.makedirs(os.path.dirname(vllm_cache_old), exist_ok=True)
        try:
            os.symlink(workspace_vllm_cache, vllm_cache_old)
        except Exception:
            pass

setup_cache_symlinks()

# ============================================================
# 모델 관리 함수
# ============================================================

def delete_model_cache(hf_model: str) -> bool:
    """HuggingFace 모델 캐시 삭제"""
    try:
        hub_cache = os.path.join(workspace_cache_dir, "hub")
        model_dir = f"models--{hf_model.replace('/', '--')}"
        path = os.path.join(hub_cache, model_dir)
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False
    except Exception:
        return False


def load_model(
    hf_model: str,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> LLM:
    """vLLM 모델 로드"""
    params = {
        "model": hf_model,
        "tensor_parallel_size": torch.cuda.device_count(),
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "dtype": "auto",
    }
    if max_model_len is not None:
        params["max_model_len"] = max_model_len

    return LLM(**params)

# ============================================================
# 프롬프트 생성
# ============================================================

def create_prompt(question: str, A: str, B: str, C: str, D: str, E: str) -> str:
    return f"""다음 객관식 질문에 답하세요.

질문:
{question}

선택지:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""

# ============================================================
# 후처리 로직
# ============================================================
def extract_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    text = text.strip()

    # 명시적 정답
    m = re.search(
        r"(?:정답|답|FINAL\s*ANSWER|ANSWER|FINAL)\s*(?:은|:|-)?\s*([ABCDE])(?=\s|[가-힣]|[.,)\]]|$)",
        text,
        re.IGNORECASE
    )
    if m:
        print(f"명시적 정답: {m.group(1)}")
        return m.group(1)

    # 단일 문자 라인
    for line in reversed(text.splitlines()[-5:]):
        m = re.match(r"^\s*([ABCDE])\s*[\.\)]?\s*$", line.strip())
        if m:
            print(f"단일 문자 라인: {m.group(1)}")
            return m.group(1)

    # 한글 보기
    kor_map = {"가":"A","나":"B","다":"C","라":"D","마":"E"}
    kor = re.search(r"\b([가나다라마])\b", text)
    if kor:
        print(f"한글 보기: {kor.group(1)}")
        return kor_map[kor.group(1)]

    return None

# ============================================================
# 단일 질문 생성
# ============================================================

def generate_answer_single(
    model: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    retry: int = 1,
) -> Tuple[Optional[str], str]:

    last_raw = ""

    for _ in range(retry + 1):
        outputs = model.generate([prompt], sampling_params)
        raw = outputs[0].outputs[0].text
        last_raw = raw

        answer = extract_answer(raw)
        # print(f"raw: {raw}")
        print(f"answer: {answer}")
        if answer is not None:
            return answer, raw

    return None, last_raw

# ============================================================
# CSV 처리
# ============================================================

def process_csv(
    model: LLM,
    model_name: str,
    input_csv_path: str,
    output_csv_path: str,
    sampling_params: SamplingParams,
) -> None:

    df = pd.read_csv(input_csv_path)
    results = []

    print(f"\n평가 모델: {model_name}")
    print(f"총 {len(df)}개 질문 처리 시작\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"처리 중 ({model_name})"):
        prompt = create_prompt(
            row["question"],
            row["A"], row["B"], row["C"], row["D"], row["E"]
        )

        answer, raw = generate_answer_single(
            model, prompt, sampling_params, retry=1
        )

        results.append({
            "id": row["id"],
            "category": row["category"],
            "sub_category": row["sub_category"],
            "level": row["level"],
            "has_table": row.get("has_table"),
            "has_fomula": row.get("has_fomula"),
            "question": row["question"],
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "D": row["D"],
            "E": row["E"],
            "gold": row["gold"],
            "answer": answer,
            "outputs_text": raw,
        })

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    pd.DataFrame(results).to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"✓ 결과 저장 완료: {output_csv_path}")

# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":

    TARGET_MODEL = "openai/gpt-oss-120b"

    DATA_DIR = "/workspace/Fin-Ben/_datasets/0_integration"
    BENCHMARK_LIST = ["1_fin_knowledge.csv"]

    # 출력 디렉토리: eval/_results/
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/1_fin_knowledge")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=128,   # 충분히 주고 후처리
        seed=2025,
    )

    print("=" * 60)
    print("gpt-oss-120b MCQ Evaluation (vLLM)")
    print("=" * 60)

    model = load_model(
        hf_model=TARGET_MODEL,
        gpu_memory_utilization=0.9,
        max_model_len=32768,
    )

    for benchmark in BENCHMARK_LIST:
        output_csv = os.path.join(
            OUTPUT_DIR,
            f"{benchmark.replace('.csv','')}_gpt-oss-120b_response.csv"
        )

        process_csv(
            model=model,
            model_name="gpt-oss-120b",
            input_csv_path=os.path.join(DATA_DIR, benchmark),
            output_csv_path=output_csv,
            sampling_params=sampling_params,
        )

    # 메모리 정리
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # delete_model_cache(MODEL_NAME)

    print("\n모든 평가 완료")
