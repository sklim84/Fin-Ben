"""
Response CSV 파일들의 category 컬럼을 통합된 카테고리로 업데이트합니다.

preprocess.py에서 적용한 카테고리 재분류 및 통합 로직을 
eval/_results/ 디렉토리의 모든 *_response.csv 파일에 적용합니다.
"""

import pandas as pd
import os
import glob
from pathlib import Path


def reclassify_categories(df):
    """
    경제학 관련 카테고리를 재분류합니다.
    
    재분류 규칙:
    1. '경제학' 카테고리의 sub_category를 기반으로 메인 카테고리로 승격
       - sub_category='거시경제학' → category='거시경제학'
       - sub_category='국제경제학' → category='국제경제학'
       - sub_category='국제' → category='국제경제학'
       - sub_category='화폐금융' → category='화폐금융'
    2. 독립 카테고리 통합
       - '미시경제' → '미시경제학'
       - '통화량조절' → '화폐금융'
    
    Args:
        df: 재분류할 DataFrame
    
    Returns:
        재분류된 DataFrame
    """
    df = df.copy()
    
    # sub_category가 NaN인 경우 빈 문자열로 처리
    if 'sub_category' in df.columns:
        df['sub_category'] = df['sub_category'].fillna('').astype(str).str.strip()
    else:
        df['sub_category'] = ''
    
    # 1단계: '경제학' 카테고리 재분류 (sub_category 기반)
    mask_econ = df['category'] == '경제학'
    reclassify_rules = {
        '거시경제학': ['거시경제학'],
        '국제경제학': ['국제경제학', '국제'],
        '화폐금융': ['화폐금융']
    }
    
    for new_cat, sub_cats in reclassify_rules.items():
        mask = mask_econ & df['sub_category'].isin(sub_cats)
        df.loc[mask, 'category'] = new_cat
        if 'sub_category' in df.columns:
            df.loc[mask, 'sub_category'] = ''
    
    # 2단계: 독립 카테고리 통합
    simple_merges = {
        '미시경제': '미시경제학',
        '통화량조절': '화폐금융'
    }
    for old_cat, new_cat in simple_merges.items():
        mask = df['category'] == old_cat
        df.loc[mask, 'category'] = new_cat
    
    return df


def merge_categories(df):
    """
    카테고리를 통합합니다.
    
    통합 규칙:
    1단계 (강력 추천):
       - '금융의 기초개념' → '금융의 기초'
       - '선물옵션' → '파생상품'
       - '금융상품 선택기준' → '금융상품'
    
    2단계 (검토 권장):
       - '디지털 자산' → '디지털 금융'
       - '탈 중앙금융' → '디지털 금융'
       - '금융시장 법률' → '금융기관'
    
    3단계 (선택적):
       - '한국은행의 기능' → '화폐금융'
       - '금리' → '화폐금융'
    
    Args:
        df: 통합할 DataFrame
    
    Returns:
        통합된 DataFrame
    """
    df = df.copy()
    
    # 통합 매핑 정의 (단계별로 그룹화)
    merge_mappings = {
        '1단계: 강력 추천 통합': {
            '금융의 기초개념': '금융의 기초',
            '선물옵션': '파생상품',
            '금융상품 선택기준': '금융상품'
        },
        '2단계: 검토 권장 통합': {
            '디지털 자산': '디지털 금융',
            '탈 중앙금융': '디지털 금융',
            '금융시장 법률': '금융기관'
        },
        '3단계: 선택적 통합': {
            '한국은행의 기능': '화폐금융',
            '금리': '화폐금융'
        }
    }
    
    # 통합 수행
    for mappings in merge_mappings.values():
        for old_cat, new_cat in mappings.items():
            mask = df['category'] == old_cat
            df.loc[mask, 'category'] = new_cat
    
    return df


def update_response_file(file_path, results_dir, outdated_dir):
    """
    Response CSV 파일의 category를 업데이트합니다.
    
    Args:
        file_path: 업데이트할 CSV 파일 경로
        results_dir: results 디렉토리 경로 (상대 경로 계산용)
        outdated_dir: 백업 파일을 저장할 outdated 디렉토리 경로
    
    Returns:
        업데이트된 행 수 (변경된 category 수)
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        if 'category' not in df.columns:
            print(f"  ⚠ 경고: 'category' 컬럼이 없습니다. 건너뜁니다.")
            return 0
        
        # 원본 카테고리 백업 (통계용)
        original_categories = df['category'].value_counts().to_dict()
        
        # 카테고리 재분류 및 통합 적용
        df = reclassify_categories(df)
        df = merge_categories(df)
        
        # 업데이트된 카테고리 통계
        updated_categories = df['category'].value_counts().to_dict()
        
        # 변경 사항 확인
        changed = False
        for old_cat, count in original_categories.items():
            if old_cat not in updated_categories or updated_categories[old_cat] != count:
                changed = True
                break
        
        # 변경된 경우에만 저장
        if changed:
            # 백업 파일 경로 생성 (outdated 디렉토리에 1_fin_knowledge 구조 유지)
            file_name = os.path.basename(file_path)
            backup_path = os.path.join(outdated_dir, "1_fin_knowledge", file_name)
            
            # 백업 디렉토리 생성
            backup_dir = os.path.dirname(backup_path)
            os.makedirs(backup_dir, exist_ok=True)
            
            # 백업 파일이 없으면 생성
            if not os.path.exists(backup_path):
                # 원본 파일을 백업으로 복사
                original_df = pd.read_csv(file_path, encoding='utf-8-sig')
                original_df.to_csv(backup_path, index=False, encoding='utf-8-sig')
            
            # 업데이트된 파일 저장
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            # 변경 통계 출력
            print(f"  ✓ 업데이트 완료")
            print(f"    - 원본 카테고리 수: {len(original_categories)}개")
            print(f"    - 업데이트 후 카테고리 수: {len(updated_categories)}개")
            print(f"    - 백업 저장: {os.path.relpath(backup_path, outdated_dir)}")
            
            return len(df)
        else:
            print(f"  - 변경 사항 없음 (이미 통합된 상태)")
            return 0
            
    except Exception as e:
        print(f"  ✗ 오류 발생: {str(e)}")
        return 0


def find_response_files(results_dir):
    """
    results_dir 디렉토리에서 모든 *_response.csv 파일을 찾습니다.
    
    Args:
        results_dir: 검색할 디렉토리 경로
    
    Returns:
        찾은 파일 경로 리스트
    """
    pattern = os.path.join(results_dir, "*_response.csv")
    files = glob.glob(pattern)
    return sorted(files)


if __name__ == "__main__":
    """
    eval/_results/1_fin_knowledge 디렉토리의 모든 response CSV 파일의 
    category 컬럼을 통합된 카테고리로 업데이트합니다.
    """
    # 스크립트 디렉토리 기준으로 results 디렉토리 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "_results", "1_fin_knowledge")
    outdated_dir = os.path.join(script_dir, "_outdated")
    
    # outdated 디렉토리 생성
    os.makedirs(outdated_dir, exist_ok=True)
    
    print("=" * 80)
    print("Response CSV 파일 카테고리 업데이트")
    print("=" * 80)
    print(f"\n검색 디렉토리: {results_dir}")
    print(f"백업 디렉토리: {outdated_dir}")
    
    # Response 파일 찾기
    response_files = find_response_files(results_dir)
    
    print(f"\n총 {len(response_files)}개 파일 발견")
    print("-" * 80)
    
    # 각 파일 처리
    total_updated = 0
    total_files = 0
    
    for file_path in response_files:
        file_name = os.path.basename(file_path)
        rel_path = os.path.relpath(file_path, results_dir)
        
        print(f"\n처리 중: {rel_path}")
        
        updated_count = update_response_file(file_path, results_dir, outdated_dir)
        
        if updated_count > 0:
            total_updated += 1
        total_files += 1
    
    # 최종 통계
    print("\n" + "=" * 80)
    print("처리 완료")
    print("=" * 80)
    print(f"  - 총 파일 수: {total_files}개")
    print(f"  - 업데이트된 파일 수: {total_updated}개")
    print(f"  - 변경 없음: {total_files - total_updated}개")
    print(f"\n백업 파일은 {outdated_dir} 디렉토리에 저장되었습니다.")
    print("=" * 80)
