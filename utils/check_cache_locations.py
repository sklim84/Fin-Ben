"""
HuggingFace 및 vLLM 캐시 디렉토리 확인 스크립트

workspace 이외의 디렉토리에 저장되는 정보를 찾습니다.
"""

import os
import subprocess
from pathlib import Path


def get_huggingface_cache_dirs():
    """HuggingFace 캐시 디렉토리 확인"""
    cache_dirs = {}
    
    # 환경 변수 확인
    env_vars = {
        "HF_HOME": os.getenv("HF_HOME"),
        "HUGGINGFACE_HUB_CACHE": os.getenv("HUGGINGFACE_HUB_CACHE"),
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
        "HF_DATASETS_CACHE": os.getenv("HF_DATASETS_CACHE"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),  # 토큰은 경로가 아니지만 확인용
    }
    
    # 기본 홈 디렉토리 캐시
    home_dir = os.path.expanduser("~")
    default_cache = os.path.join(home_dir, ".cache", "huggingface")
    
    cache_dirs["환경 변수"] = env_vars
    cache_dirs["기본 홈 디렉토리 캐시"] = default_cache
    
    # 실제 사용되는 캐시 디렉토리
    actual_cache = env_vars.get("HUGGINGFACE_HUB_CACHE") or os.path.join(default_cache, "hub")
    cache_dirs["실제 모델 캐시"] = actual_cache
    
    return cache_dirs


def get_vllm_cache_dirs():
    """vLLM 캐시 디렉토리 확인"""
    cache_dirs = {}
    
    # vLLM torch.compile 캐시
    home_dir = os.path.expanduser("~")
    vllm_cache = os.path.join(home_dir, ".cache", "vllm")
    cache_dirs["vLLM 캐시 (torch.compile)"] = vllm_cache
    
    # 환경 변수 확인
    vllm_cache_env = os.getenv("VLLM_CACHE_DIR")
    if vllm_cache_env:
        cache_dirs["VLLM_CACHE_DIR 환경 변수"] = vllm_cache_env
    
    return cache_dirs


def get_pytorch_cache_dirs():
    """PyTorch 캐시 디렉토리 확인"""
    cache_dirs = {}
    
    # PyTorch 홈 디렉토리
    home_dir = os.path.expanduser("~")
    torch_cache = os.path.join(home_dir, ".cache", "torch")
    cache_dirs["PyTorch 캐시"] = torch_cache
    
    # 환경 변수 확인
    torch_home = os.getenv("TORCH_HOME")
    if torch_home:
        cache_dirs["TORCH_HOME 환경 변수"] = torch_home
    
    return cache_dirs


def check_directory_size(path):
    """디렉토리 크기 확인"""
    if not os.path.exists(path):
        return "존재하지 않음", 0
    
    try:
        result = subprocess.run(
            ["du", "-sh", path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            return "존재", size
        else:
            return "존재", "크기 확인 실패"
    except Exception as e:
        return "존재", f"크기 확인 오류: {e}"


def check_workspace_mount():
    """workspace 마운트 포인트 확인"""
    workspace_path = "/workspace"
    
    try:
        # df 명령어로 마운트 포인트 확인
        result = subprocess.run(
            ["df", "-h", workspace_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                mount_info = lines[1].split()
                return {
                    "파일시스템": mount_info[0],
                    "크기": mount_info[1],
                    "사용": mount_info[2],
                    "사용률": mount_info[4],
                    "마운트포인트": mount_info[5]
                }
    except Exception as e:
        return {"오류": str(e)}
    
    return None


def main():
    """메인 함수"""
    print("=" * 80)
    print("HuggingFace 및 vLLM 캐시 디렉토리 확인")
    print("=" * 80)
    
    # workspace 마운트 정보
    print("\n[1] Workspace 마운트 정보")
    print("-" * 80)
    mount_info = check_workspace_mount()
    if mount_info:
        for key, value in mount_info.items():
            print(f"  {key}: {value}")
    else:
        print("  마운트 정보를 확인할 수 없습니다.")
    
    # HuggingFace 캐시 디렉토리
    print("\n[2] HuggingFace 캐시 디렉토리")
    print("-" * 80)
    hf_dirs = get_huggingface_cache_dirs()
    for category, paths in hf_dirs.items():
        if isinstance(paths, dict):
            print(f"\n  {category}:")
            for key, value in paths.items():
                if value:
                    status, size = check_directory_size(value)
                    print(f"    {key}: {value} ({status}, {size})")
                else:
                    print(f"    {key}: 설정되지 않음")
        else:
            status, size = check_directory_size(paths)
            print(f"  {category}: {paths} ({status}, {size})")
    
    # vLLM 캐시 디렉토리
    print("\n[3] vLLM 캐시 디렉토리")
    print("-" * 80)
    vllm_dirs = get_vllm_cache_dirs()
    for category, path in vllm_dirs.items():
        status, size = check_directory_size(path)
        print(f"  {category}: {path} ({status}, {size})")
    
    # PyTorch 캐시 디렉토리
    print("\n[4] PyTorch 캐시 디렉토리")
    print("-" * 80)
    torch_dirs = get_pytorch_cache_dirs()
    for category, path in torch_dirs.items():
        status, size = check_directory_size(path)
        print(f"  {category}: {path} ({status}, {size})")
    
    # workspace 외부 디렉토리 확인
    print("\n[5] Workspace 외부 디렉토리 확인")
    print("-" * 80)
    workspace_path = "/workspace"
    all_dirs = []
    
    # 모든 캐시 디렉토리 수집
    for category, paths in hf_dirs.items():
        if isinstance(paths, dict):
            for value in paths.values():
                if value and isinstance(value, str) and os.path.isabs(value):
                    all_dirs.append(value)
        elif isinstance(paths, str) and os.path.isabs(paths):
            all_dirs.append(paths)
    
    for category, path in vllm_dirs.items():
        if isinstance(path, str) and os.path.isabs(path):
            all_dirs.append(path)
    
    for category, path in torch_dirs.items():
        if isinstance(path, str) and os.path.isabs(path):
            all_dirs.append(path)
    
    # workspace 외부 디렉토리 필터링
    external_dirs = []
    for dir_path in set(all_dirs):
        if dir_path and not dir_path.startswith(workspace_path):
            status, size = check_directory_size(dir_path)
            external_dirs.append((dir_path, status, size))
    
    if external_dirs:
        print("  ⚠️  Workspace 외부에 저장되는 디렉토리:")
        for dir_path, status, size in external_dirs:
            print(f"    - {dir_path} ({status}, {size})")
    else:
        print("  ✓ Workspace 외부에 저장되는 디렉토리가 없습니다.")
    
    # 권장 사항
    print("\n[6] 권장 사항")
    print("-" * 80)
    if external_dirs:
        print("  Workspace 외부 디렉토리를 workspace로 이동하려면:")
        print("  1. 환경 변수를 설정하여 캐시 위치 변경")
        print("  2. 또는 심볼릭 링크 생성")
        print("\n  예시:")
        print('    export HF_HOME="/workspace/.cache/huggingface"')
        print('    export HUGGINGFACE_HUB_CACHE="/workspace/.cache/huggingface/hub"')
        print('    export TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers"')
        print('    export HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets"')
    else:
        print("  모든 캐시가 workspace 내에 저장되고 있습니다.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

