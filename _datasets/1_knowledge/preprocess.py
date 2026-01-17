import pandas as pd
import os

def _get_category_counts(df, categories):
    """
    카테고리별 문항 수를 계산합니다.
    
    Args:
        df: DataFrame
        categories: 카운트할 카테고리 리스트
    
    Returns:
        카테고리별 문항 수 딕셔너리
    """
    return {
        cat: len(df[df['category'] == cat])
        for cat in categories
        if len(df[df['category'] == cat]) > 0
    }

def _print_reclassification_stats(before_stats, after_stats, simple_merges):
    """
    재분류 통계를 출력합니다.
    
    Args:
        before_stats: 재분류 전 통계 딕셔너리
        after_stats: 재분류 후 통계 딕셔너리
        simple_merges: 통합 규칙 딕셔너리
    """
    if not before_stats:
        return
    
    print(f"\n{'='*60}")
    print("카테고리 재분류 수행")
    print(f"{'='*60}")
    
    if '경제학' in before_stats:
        print(f"  - '경제학' ({before_stats['경제학']}개) → 재분류 완료")
    
    for old_cat, new_cat in simple_merges.items():
        if old_cat in before_stats:
            print(f"  - '{old_cat}' ({before_stats[old_cat]}개) → '{new_cat}'로 통합")
    
    if after_stats:
        print(f"\n  재분류 결과:")
        for cat, count in sorted(after_stats.items()):
            print(f"    - {cat}: {count}개")

def _print_merge_stats(merge_before, merge_after, merge_mappings):
    """
    통합 통계를 출력합니다.
    
    Args:
        merge_before: 통합 전 통계 딕셔너리
        merge_after: 통합 후 통계 딕셔너리
        merge_mappings: 통합 매핑 딕셔너리
    """
    if not merge_before:
        return
    
    print(f"\n{'='*60}")
    print("카테고리 통합 수행")
    print(f"{'='*60}")
    
    current_step = None
    for old_cat in sorted(merge_before.keys(), 
                          key=lambda x: list(merge_mappings.keys()).index(merge_before[x]['step'])):
        info = merge_before[old_cat]
        if info['step'] != current_step:
            current_step = info['step']
            print(f"\n  [{current_step}]")
        print(f"    - '{old_cat}' ({info['count']}개) → '{info['target']}'로 통합")
    
    if merge_after:
        print(f"\n  통합 결과:")
        for cat, count in sorted(merge_after.items()):
            print(f"    - {cat}: {count}개")

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
    df['sub_category'] = df['sub_category'].fillna('').astype(str).str.strip()
    
    # 재분류 전 통계
    categories_to_track = ['경제학', '미시경제', '통화량조절']
    before_stats = _get_category_counts(df, categories_to_track)
    
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
        df.loc[mask, 'sub_category'] = ''
    
    # 2단계: 독립 카테고리 통합
    simple_merges = {
        '미시경제': '미시경제학',
        '통화량조절': '화폐금융'
    }
    for old_cat, new_cat in simple_merges.items():
        mask = df['category'] == old_cat
        df.loc[mask, 'category'] = new_cat
    
    # 재분류 후 통계 및 출력
    new_categories = ['거시경제학', '국제경제학', '화폐금융', '미시경제학']
    after_stats = _get_category_counts(df, new_categories)
    _print_reclassification_stats(before_stats, after_stats, simple_merges)
    
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
    
    # 통합 전 통계 수집
    merge_before = {}
    for step_name, mappings in merge_mappings.items():
        for old_cat, new_cat in mappings.items():
            count = len(df[df['category'] == old_cat])
            if count > 0:
                merge_before[old_cat] = {'count': count, 'target': new_cat, 'step': step_name}
    
    # 통합 수행
    for mappings in merge_mappings.values():
        for old_cat, new_cat in mappings.items():
            mask = df['category'] == old_cat
            df.loc[mask, 'category'] = new_cat
    
    # 통합 후 통계 수집 및 출력
    merged_categories = set()
    for mappings in merge_mappings.values():
        merged_categories.update(mappings.values())
    
    merge_after = _get_category_counts(df, merged_categories)
    _print_merge_stats(merge_before, merge_after, merge_mappings)
    
    return df

def load_and_prepare_csv_files(base_dir, csv_files):
    """
    CSV 파일들을 로드하고 전처리합니다.
    
    Args:
        base_dir: CSV 파일이 있는 디렉토리 경로
        csv_files: 로드할 CSV 파일명 리스트
    
    Returns:
        전처리된 DataFrame 리스트
    """
    dataframes = []
    standard_columns = ['id', 'category', 'sub_category', 'level', 'has_table', 
                       'has_fomula', 'question', 'A', 'B', 'C', 'D', 'E', 'gold']
    
    for csv_file in csv_files:
        file_path = os.path.join(base_dir, csv_file)
        
        if not os.path.exists(file_path):
            print(f"⚠ 경고: 파일을 찾을 수 없습니다 - {csv_file}")
            continue
        
        print(f"\n처리 중: {csv_file}")
        df = pd.read_csv(file_path)
        print(f"  - 행 수: {len(df)}")
        print(f"  - 컬럼: {', '.join(df.columns.tolist())}")
        
        # sub_category 컬럼이 없으면 추가
        if 'sub_category' not in df.columns:
            print(f"  - sub_category 컬럼 추가 (NaN 값으로 채움)")
            df['sub_category'] = None
        
        # 컬럼 순서 통일
        df = df[[col for col in standard_columns if col in df.columns]]
        dataframes.append(df)
        print(f"  ✓ 처리 완료")
    
    return dataframes

def save_merged_dataframe(df, base_dir, filename="1_fin_knowledge.csv"):
    """
    통합된 DataFrame을 CSV 파일로 저장합니다.
    
    Args:
        df: 저장할 DataFrame
        base_dir: 현재 스크립트 디렉토리
        filename: 저장할 파일명
    
    Returns:
        저장된 파일 경로
    """
    output_dir = os.path.join(os.path.dirname(base_dir), "0_integration")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path

if __name__ == "__main__":
    """
    세 개의 CSV 파일을 통합하고 카테고리를 재분류/통합합니다.
    
    처리 과정:
    1. CSV 파일 통합 및 전처리
    2. 카테고리 재분류 (경제학 관련)
    3. 카테고리 통합 (1~3단계)
    4. ID 재설정 및 저장
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [
        "(FN)경영학_97문항.csv",
        "(FN)경제학_99문항.csv",
        "(FN)금융시사상식_100문항.csv"
    ]
    
    print("=" * 60)
    print("CSV 파일 통합 시작")
    print("=" * 60)
    
    # CSV 파일 로드 및 전처리
    dataframes = load_and_prepare_csv_files(base_dir, csv_files)
    
    if not dataframes:
        print("\n✗ 통합할 데이터가 없습니다.")
        exit(1)
    
    # 데이터프레임 통합
    print(f"\n{'='*60}")
    print("데이터 통합 중...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"통합 완료:")
    print(f"  - 총 행 수: {len(merged_df)}")
    print(f"  - 컬럼: {', '.join(merged_df.columns.tolist())}")
    
    # 카테고리 재분류 및 통합
    merged_df = reclassify_categories(merged_df)
    merged_df = merge_categories(merged_df)
    
    # ID 재설정 및 저장
    merged_df['id'] = range(1, len(merged_df) + 1)
    output_path = save_merged_dataframe(merged_df, base_dir)
    
    print(f"\n✓ 통합 파일 저장 완료: {output_path}")
    print("=" * 60)
    
