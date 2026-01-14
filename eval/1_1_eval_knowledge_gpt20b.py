"""
LLM 모델 평가 스크립트 (GPT-OSS 모델용, Transformers 직접 사용)

- 금융지식(MCQ) 벤치마크 전용
- 출력은 반드시 A~E 중 하나의 알파벳만 허용
- LogitsProcessor로 토큰 레벨 제약 적용
- Multi-GPU(device_map="auto") 환경에서 device mismatch 방지
"""

import os
import gc
import shutil
import traceback
from typing import Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

# ============================================================
# 캐시 설정
# ============================================================
workspace_cache_dir = "/workspace/.cache/huggingface"
os.makedirs(workspace_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = workspace_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(workspace_cache_dir, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(workspace_cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(workspace_cache_dir, "datasets")

# PyTorch CUDA 메모리 단편화 완화
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# accelerate CPU offload 비활성화 (MoE 모델 호환성)
os.environ.setdefault("ACCELERATE_DISABLE_CPU_OFFLOAD", "1")

# vLLM 캐시 설정
workspace_vllm_cache = "/workspace/.cache/vllm"
os.makedirs(workspace_vllm_cache, exist_ok=True)


def setup_cache_symlinks():
    """workspace 캐시 디렉토리로 심볼릭 링크 생성"""
    home_cache = os.path.expanduser("~/.cache")
    
    # HuggingFace 캐시 심볼릭 링크
    hf_cache_old = os.path.join(home_cache, "huggingface")
    if os.path.exists(hf_cache_old) and not os.path.islink(hf_cache_old):
        print(f"경고: {hf_cache_old} 디렉토리가 존재합니다. 수동으로 삭제해주세요.")
    elif not os.path.exists(hf_cache_old):
        try:
            os.makedirs(os.path.dirname(hf_cache_old), exist_ok=True)
            os.symlink(workspace_cache_dir, hf_cache_old)
            print(f"✓ HuggingFace 캐시 심볼릭 링크 생성: {hf_cache_old} -> {workspace_cache_dir}")
        except Exception as e:
            print(f"HuggingFace 캐시 심볼릭 링크 생성 중 오류 (무시): {e}")
    
    # vLLM 캐시 심볼릭 링크
    vllm_cache_old = os.path.join(home_cache, "vllm")
    if os.path.exists(vllm_cache_old) and not os.path.islink(vllm_cache_old):
        print(f"경고: {vllm_cache_old} 디렉토리가 존재합니다. 수동으로 삭제해주세요.")
    elif not os.path.exists(vllm_cache_old):
        try:
            os.makedirs(os.path.dirname(vllm_cache_old), exist_ok=True)
            os.symlink(workspace_vllm_cache, vllm_cache_old)
            print(f"✓ vLLM 캐시 심볼릭 링크 생성: {vllm_cache_old} -> {workspace_vllm_cache}")
        except Exception as e:
            print(f"vLLM 캐시 심볼릭 링크 생성 중 오류 (무시): {e}")


setup_cache_symlinks()


# ============================================================
# LogitsProcessor: A–E만 허용
# ============================================================
class OnlyABCDELogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: AutoTokenizer):
        token_ids = tokenizer(["A", "B", "C", "D", "E"], add_special_tokens=False)["input_ids"]
        self.allowed_token_ids = set()
        for t in token_ids:
            if len(t) == 1:
                self.allowed_token_ids.add(t[0])

        if len(self.allowed_token_ids) < 5:
            raise ValueError(
                f"Tokenizer does not map all of A-E to single tokens. allowed={self.allowed_token_ids}"
            )

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0.0
        return scores + mask

def delete_model_cache(hf_model: str) -> bool:
    """모델 캐시 삭제 (옵션)"""
    try:
        hub_cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(workspace_cache_dir, "hub"))
        model_dir_name = f"models--{hf_model.replace('/', '--')}"
        model_cache_path = os.path.join(hub_cache_dir, model_dir_name)

        if os.path.exists(model_cache_path):
            print(f"\n모델 캐시 삭제 중: {model_cache_path}")
            shutil.rmtree(model_cache_path)
            print(f"✓ 모델 캐시 삭제 완료: {hf_model}")
            return True
        else:
            print(f"모델 캐시를 찾을 수 없습니다: {model_cache_path}")
            return False
    except Exception as e:
        print(f"모델 캐시 삭제 중 오류 발생: {e}")
        return False


# ============================================================
# 모델 로딩
# ============================================================
def load_model(hf_model: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    모델 로딩 함수
    
    Args:
        hf_model: HuggingFace 모델 경로
    
    Returns:
        (모델, 토크나이저) 튜플
    """
    print(f"\n모델 로딩 중: {hf_model}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("✓ 모델 로딩 완료")
    return model, tokenizer


# ============================================================
# 프롬프트 생성
# ============================================================
def create_prompt(question: str, A: str, B: str, C: str, D: str, E: str) -> str:
    """객관식 질문 프롬프트 생성"""
    return f"""다음 객관식 질문에 답하세요.

질문:
{question}

선택지:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

정답은 A, B, C, D, E 중 하나만 출력하세요.
"""


# ============================================================
# 단일 질문 생성 (device mismatch 방지)
# ============================================================
@torch.no_grad()
def generate_answer_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    temperature: float = 0.0,
) -> Tuple[Optional[str], Optional[str]]:
    """
    단일 질문에 대한 답변 생성
    
    Args:
        model: 모델 객체
        tokenizer: 토크나이저 객체
        prompt: 프롬프트 문자열
        temperature: 생성 온도
    
    Returns:
        (답변 알파벳, 답변 텍스트) 튜플
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # 분산 모델에서는 임베딩 레이어의 디바이스에 입력을 맞춤
        if torch.cuda.is_available():
            embed_device = model.get_input_embeddings().weight.device
            inputs = {k: v.to(embed_device) for k, v in inputs.items()}

        logits_processor = LogitsProcessorList([OnlyABCDELogitsProcessor(tokenizer)])

        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=temperature,
            do_sample=False,
            logits_processor=logits_processor,
        )

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][input_len:]
        answer_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        print(f"answer_text: {answer_text}")

        if answer_text:
            answer = answer_text.upper()[0]
            if answer in ["A", "B", "C", "D", "E"]:
                return answer, answer_text

        return None, answer_text

    except Exception as e:
        print(f"생성 오류: {e}")
        print(f"오류 타입: {type(e).__name__}")
        traceback.print_exc()
        return None, None


# ============================================================
# CSV 처리
# ============================================================
def process_csv(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    input_csv_path: str,
    output_csv_path: str,
) -> None:
    """
    CSV 파일을 읽어 모델 평가 수행
    
    Args:
        model: 모델 객체
        tokenizer: 토크나이저 객체
        model_name: 모델 이름
        input_csv_path: 입력 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로
    """
    data = pd.read_csv(input_csv_path)
    results = []

    print(f"\n평가 모델: {model_name}")
    print(f"총 {len(data)}개 질문 처리 시작\n")

    for _, row in tqdm(data.iterrows(), total=len(data)):
        prompt = create_prompt(row["question"], row["A"], row["B"], row["C"], row["D"], row["E"])
        answer, answer_text = generate_answer_single(model, tokenizer, prompt)

        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "sub_category": row["sub_category"],
                "level": row["level"],
                "question": row["question"],
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "E": row["E"],
                "gold": row["gold"],
                "answer": answer,
                "answer_text": answer_text,
            }
        )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ 결과 저장 완료: {output_csv_path}")


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    TARGET_MODELS = [
        "openai/gpt-oss-20b",
    ]

    file_path = "/workspace/Fin-Ben/_datasets/0_integration"
    benchmark_list = ["1_fin_knowledge.csv"]

    for hf_model in TARGET_MODELS:
        model_name = hf_model.split("/")[-1]

        model, tokenizer = load_model(hf_model)

        for benchmark in benchmark_list:
            input_csv_path = os.path.join(file_path, benchmark)

            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/1_fin_knowledge")
            os.makedirs(results_dir, exist_ok=True)

            output_csv_path = os.path.join(
                results_dir,
                f"{benchmark.replace('.csv', '')}_{model_name_safe}_response.csv",
            )

            process_csv(model, tokenizer, model_name, input_csv_path, output_csv_path)

        # 메모리 해제
        del model
        torch.cuda.empty_cache()  # GPU 메모리 정리

        # 디스크 관리 (옵션)
        # delete_model_cache(hf_model)

    print("\n모든 평가 완료")

