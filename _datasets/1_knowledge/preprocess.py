import pandas as pd
import os

def merge_csv_files():
    """
    세 개의 CSV 파일을 통합합니다.
    - (FN)경영학_97문항.csv
    - (FN)경제학_99문항.csv
    - (FN)금융시사상식_100문항.csv
    
    금융시사상식 파일에는 sub_category 컬럼이 없으므로 추가합니다.
    """
    # 현재 스크립트가 있는 디렉토리 경로
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CSV 파일 경로
    csv_files = [
        "(FN)경영학_97문항.csv",
        "(FN)경제학_99문항.csv",
        "(FN)금융시사상식_100문항.csv"
    ]
    
    # 통합할 데이터프레임 리스트
    dataframes = []
    
    print("=" * 60)
    print("CSV 파일 통합 시작")
    print("=" * 60)
    
    for csv_file in csv_files:
        file_path = os.path.join(base_dir, csv_file)
        
        if not os.path.exists(file_path):
            print(f"⚠ 경고: 파일을 찾을 수 없습니다 - {csv_file}")
            continue
        
        print(f"\n처리 중: {csv_file}")
        df = pd.read_csv(file_path)
        print(f"  - 행 수: {len(df)}")
        print(f"  - 컬럼: {', '.join(df.columns.tolist())}")
        
        # 금융시사상식 파일에 sub_category 컬럼이 없으면 추가
        if 'sub_category' not in df.columns:
            print(f"  - sub_category 컬럼 추가 (NaN 값으로 채움)")
            df['sub_category'] = None
        
        # 컬럼 순서 통일 (표준 순서)
        standard_columns = ['id', 'category', 'sub_category', 'level', 'has_table', 
                           'has_fomula', 'question', 'A', 'B', 'C', 'D', 'E', 'gold']
        
        # 컬럼 순서 재정렬 (존재하는 컬럼만)
        df = df[[col for col in standard_columns if col in df.columns]]
        
        dataframes.append(df)
        print(f"  ✓ 처리 완료")
    
    if not dataframes:
        print("\n✗ 통합할 데이터가 없습니다.")
        return
    
    # 모든 데이터프레임 통합
    print(f"\n{'='*60}")
    print("데이터 통합 중...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # ID 재설정 (1부터 시작)
    merged_df['id'] = range(1, len(merged_df) + 1)
    
    print(f"통합 완료:")
    print(f"  - 총 행 수: {len(merged_df)}")
    print(f"  - 컬럼: {', '.join(merged_df.columns.tolist())}")
    
    # 통합된 파일 저장 (0_integration 디렉토리에 저장)
    datasets_dir = os.path.dirname(base_dir)  # _datasets/ 디렉토리
    output_dir = os.path.join(datasets_dir, "0_integration")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "1_fin_knowledge.csv")
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ 통합 파일 저장 완료: {output_path}")
    print("=" * 60)
    
    return merged_df

if __name__ == "__main__":
    merge_csv_files()
