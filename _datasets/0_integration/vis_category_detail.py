"""
금융 벤치마크 데이터셋 카테고리 분석 및 시각화 스크립트

이 스크립트는 0_integration 디렉토리의 CSV 파일들을 읽어서
카테고리별 문항 수를 집계하고 Sunburst 차트로 시각화합니다.
"""

import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# 현재 스크립트 디렉토리
script_dir = Path(__file__).parent

# =================================
# 카테고리 번역 딕셔너리
# =================================

# 메인 카테고리 번역 (파일명 기반)
main_category_mapping = {
    '1_fin_knowledge': 'Financial Knowledge',
    '2_fin_reasoning': 'Financial Reasoning',
    '3_fin_toxicity': 'Financial Toxicity'
}

# 서브 카테고리 번역 (실제 데이터 기반)
category_translation = {
    # 회계 관련
    '중급회계': 'Intermediate Accounting',
    '세법': 'Tax Law',
    
    # 경제학 관련
    # '경제학': 'Economics',  # 재분류됨: 경제학 → 거시경제학/국제경제학/화폐금융
    '미시경제학': 'Microeconomics',
    # '미시경제': 'Microeconomics',  # 통합됨: 미시경제 → 미시경제학
    '거시경제학': 'Macroeconomics',
    '국제경제학': 'International Economics',
    '계량경제': 'Econometrics',
    
    # 재무관리 관련
    '재무관리': 'Financial Management',
    
    # 금융시장 관련
    '금융상품': 'Financial Products',
    # '금융상품 선택기준': 'Financial Product Selection',  # 통합됨: 금융상품 → 금융상품
    '금융의 기초': 'Financial Fundamentals',
    # '금융의 기초개념': 'Financial Basic Concepts',  # 통합됨: 금융의 기초개념 → 금융의 기초
    '금융기관': 'Financial Institutions',
    # '금융시장 법률': 'Financial Market Law',  # 통합됨: 금융시장 법률 → 금융기관
    # '금리': 'Interest Rate',  # 통합됨: 금리 → 화폐금융
    '화폐금융': 'Monetary Finance',
    # '통화량조절': 'Money Supply Control',  # 통합됨: 통화량조절 → 화폐금융
    # '한국은행의 기능': 'Bank of Korea Functions',  # 통합됨: 한국은행의 기능 → 화폐금융
    
    # 시장 관련
    '증권시장': 'Securities Market',
    '채권시장': 'Bond Market',
    '부동산시장': 'Real Estate Market',
    '유통시장': 'Distribution Market',
    
    # 파생상품 관련
    '파생상품': 'Derivatives',
    # '선물옵션': 'Futures & Options',  # 통합됨: 선물옵션 → 파생상품
    
    # 디지털 금융 관련
    '디지털 금융': 'Digital Finance',
    # '디지털 자산': 'Digital Assets',  # 통합됨: 디지털 자산 → 디지털 금융
    # '탈 중앙금융': 'Decentralized Finance',  # 통합됨: 탈 중앙금융 → 디지털 금융
    
    # 기타
    '생산운영관리': 'Production & Operations Management',
    '보험상품': 'Insurance Products',
    '국제금융정책': 'International Financial Policy',
    
    # Toxicity 데이터셋의 서브 카테고리
    '공포 불안 조장': 'Inciting Fear & Anxiety',
    '불법 부정행위 조언': 'Illegal Misconduct Advice',
    '정치 선동 / 여론 조작': 'Political Incitement / Opinion Manipulation',
    '허위정보 생성': 'False Information Generation',
    
    # Reasoning 데이터셋의 context 배치 유형 (이미 영어이지만 더 읽기 쉽게 변환)
    'context_relevant_dispersed': 'Relevant Info Dispersed',
    'context_relevant_front': 'Relevant Info at Front',
    'context_relevant_middle': 'Relevant Info at Middle',
    'context_relevant_end': 'Relevant Info at End',
    'context_relevant_scattered': 'Relevant Info Scattered',
    'context_relevant_only_shuffled': 'Relevant Info Only (Shuffled)',
    'context_relevant_only': 'Relevant Info Only',
    'context_relevant_middle_with_en_noise': 'Relevant Info at Middle (EN Noise)',
    
    # 빈 값 처리
    '': 'Uncategorized',
}

# =================================
# 데이터 로드 및 처리 함수
# =================================

def aggregate_csv(csv_path: str, main_category: str, filename: str = None) -> pd.DataFrame:
    """
    CSV 파일을 로드하고 카테고리별 집계 수행 (성능 최적화: 필요한 컬럼만 읽기)
    
    Args:
        csv_path: CSV 파일 경로
        main_category: 메인 카테고리 이름 (영어)
        filename: 파일명 (로직 분기를 위해 사용)
    
    Returns:
        집계된 DataFrame (Main Category, Sub Category, Count 컬럼만 포함)
    """
    try:
        # 필요한 컬럼만 읽기 (성능 최적화)
        usecols = ['category']
        
        # CSV 파일 읽기 (인코딩 시도, 필요한 컬럼만 읽기)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', usecols=usecols)
        except (UnicodeDecodeError, ValueError):
            try:
                df = pd.read_csv(csv_path, encoding='cp949', usecols=usecols)
            except (UnicodeDecodeError, ValueError):
                try:
                    df = pd.read_csv(csv_path, encoding='latin-1', usecols=usecols)
                except ValueError:
                    # usecols가 실패하면 전체 읽기 후 필요한 컬럼만 선택
                    try:
                        df = pd.read_csv(csv_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_path, encoding='cp949')
                    if 'category' not in df.columns:
                        print(f"경고: {csv_path}에 'category' 컬럼이 없습니다.")
                        return None
                    df = df[['category']]
        
        if 'category' not in df.columns:
            print(f"경고: {csv_path}에 'category' 컬럼이 없습니다.")
            return None
        
        # 서브 카테고리 번역 적용 (벡터화된 연산 사용)
        df['Sub Category'] = df['category'].map(
            lambda x: category_translation.get(x, x if pd.notna(x) else 'Uncategorized')
        )
        
        # 집계 수행 (필요한 컬럼만 사용)
        category_counts = df.groupby(['Sub Category'], observed=True).size().reset_index(name='Count')
        
        # 메인 카테고리 추가
        category_counts['Main Category'] = main_category
        
        # 컬럼 순서 정리
        category_counts = category_counts[['Main Category', 'Sub Category', 'Count']]
        
        return category_counts
    
    except Exception as e:
        print(f"오류: {csv_path} 파일을 읽는 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 처리할 CSV 파일 목록
    csv_files = [
        ('1_fin_knowledge.csv', 'Financial Knowledge'),
        ('2_fin_reasoning.csv', 'Financial Reasoning'),
        ('3_fin_toxicity.csv', 'Financial Toxicity')
    ]
    
    # 집계 결과를 담을 리스트 (DataFrame 통합 없이)
    aggregated_results = []
    
    print("=" * 60)
    print("금융 벤치마크 데이터셋 분석")
    print("=" * 60)
    
    # 각 CSV 파일을 독립적으로 집계
    for filename, main_category in csv_files:
        csv_path = script_dir / filename
        
        if not csv_path.exists():
            print(f"경고: {filename} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        print(f"처리 중: {filename}...", end=' ', flush=True)
        category_counts = aggregate_csv(str(csv_path), main_category, filename)
        
        if category_counts is not None:
            total_count = category_counts['Count'].sum()
            print(f"완료 ({total_count}개 행, {len(category_counts)}개 카테고리)")
            aggregated_results.append(category_counts)
        else:
            print("실패")
    
    if not aggregated_results:
        print("\n오류: 처리할 수 있는 데이터가 없습니다.")
        exit(1)
    
    # 집계 결과만 통합 (컬럼 구조가 동일함)
    category_counts = pd.concat(aggregated_results, ignore_index=True)
    total_rows = category_counts['Count'].sum()
    print(f"\n통합 집계 결과: 총 {total_rows}개 행 (집계된 데이터)")
    
    print("\n" + "=" * 60)
    print("카테고리별 집계 결과")
    print("=" * 60)
    print(category_counts.head(20))
    print(f"\n총 {len(category_counts)}개 카테고리 조합")
    
    # Sunburst 차트 생성
    print("\n" + "=" * 60)
    print("시각화 생성 중...")
    print("=" * 60)
    
    fig = px.sunburst(
        category_counts,
        path=['Main Category', 'Sub Category'],
        values='Count',
        color='Count',
            color_continuous_scale='RdBu',
            # title='Financial Benchmark Dataset: Category Distribution'
        )
    
    # 레이아웃 설정 (차트와 범례 간격 조정)
    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=5),  # 오른쪽 마진을 0으로 설정하여 범례와 가까워지도록
        font=dict(size=12),
        # 범례(색상 바) 위치 및 간격 조정
        coloraxis_colorbar=dict(
            xpad=5,  # 범례와 차트 사이의 간격 (기본값보다 작게)
            len=0.8,  # 범례 길이 조정
        )
    )
    
    # 결과 디렉토리 생성
    results_dir = script_dir / '_results'
    results_dir.mkdir(exist_ok=True)
    
    # HTML 저장 (빠르고 대화형)
    html_path = results_dir / 'stats.html'
    fig.write_html(str(html_path))
    print(f"✓ HTML 저장 완료: {html_path}")
    
    # 차트 표시 (선택적, 주석 처리 가능)
    # fig.show()
    
    # PDF 저장 (Kaleido 엔진 사용, Chrome 필요)
    pdf_path = results_dir / 'stats.pdf'
    
    fig.write_image(str(pdf_path), width=1200, height=1200)
    print(f"✓ PDF 저장 완료: {pdf_path}")
    
    # 집계 결과 CSV 저장
    csv_output_path = results_dir / 'category_stats.csv'
    category_counts.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 집계 결과 CSV 저장 완료: {csv_output_path}")

