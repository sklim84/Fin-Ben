"""
LLM 모델 평가 결과 통계 계산 스크립트

이 스크립트는 1_1_eval_knowledge.py에서 생성된 CSV 파일을 읽어서
정답 비교 및 통계를 계산하고 결과를 저장합니다.
"""

import pandas as pd
import os
import json
import glob
from typing import Optional, Dict


# =================================
# 정답 비교 함수
# =================================
def check_answer(predicted: Optional[str], gold: str) -> bool:
    """
    모델 답변과 정답 비교
    
    Args:
        predicted: 모델이 예측한 답변 (A~E 또는 None/NaN)
        gold: 정답 (A~E)
    
    Returns:
        정답 여부 (True/False)
    """
    # None 또는 NaN 체크 (pandas에서 읽은 NaN은 float 타입)
    if predicted is None or pd.isna(predicted):
        return False
    # 문자열로 변환 후 대소문자 구분 없이 비교
    return str(predicted).upper().strip() == str(gold).upper().strip()


# =================================
# 통계 계산 함수
# =================================
def calculate_statistics(results_df: pd.DataFrame, model_name: str) -> Dict:
    """
    평가 결과 통계 계산
    
    Args:
        results_df: 결과 DataFrame
        model_name: 모델 이름
    
    Returns:
        통계 딕셔너리
    """
    answer_col = "answer"
    is_correct_col = "is_correct"
    
    # 전체 통계
    total = len(results_df)
    correct = results_df[is_correct_col].sum()
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    stats = {
        "model": model_name,
        "total_questions": int(total),
        "correct_answers": int(correct),
        "wrong_answers": int(total - correct),
        "accuracy": round(accuracy, 2),
        "accuracy_percentage": f"{accuracy:.2f}%"
    }
    
    # 카테고리별 통계
    if 'category' in results_df.columns:
        category_stats = []
        for category in results_df['category'].unique():
            cat_df = results_df[results_df['category'] == category]
            cat_total = len(cat_df)
            cat_correct = cat_df[is_correct_col].sum()
            cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0.0
            category_stats.append({
                "category": category,
                "total": int(cat_total),
                "correct": int(cat_correct),
                "wrong": int(cat_total - cat_correct),
                "accuracy": round(cat_accuracy, 2),
                "accuracy_percentage": f"{cat_accuracy:.2f}%"
            })
        stats["by_category"] = category_stats
    
    # 레벨별 통계
    if 'level' in results_df.columns:
        level_stats = []
        for level in sorted(results_df['level'].unique()):
            level_df = results_df[results_df['level'] == level]
            level_total = len(level_df)
            level_correct = level_df[is_correct_col].sum()
            level_accuracy = (level_correct / level_total * 100) if level_total > 0 else 0.0
            level_stats.append({
                "level": level,
                "total": int(level_total),
                "correct": int(level_correct),
                "wrong": int(level_total - level_correct),
                "accuracy": round(level_accuracy, 2),
                "accuracy_percentage": f"{level_accuracy:.2f}%"
            })
        stats["by_level"] = level_stats
    
    # 서브카테고리별 통계
    if 'sub_category' in results_df.columns:
        subcat_stats = []
        for subcat in results_df['sub_category'].unique():
            subcat_df = results_df[results_df['sub_category'] == subcat]
            subcat_total = len(subcat_df)
            subcat_correct = subcat_df[is_correct_col].sum()
            subcat_accuracy = (subcat_correct / subcat_total * 100) if subcat_total > 0 else 0.0
            subcat_stats.append({
                "sub_category": subcat,
                "total": int(subcat_total),
                "correct": int(subcat_correct),
                "wrong": int(subcat_total - subcat_correct),
                "accuracy": round(subcat_accuracy, 2),
                "accuracy_percentage": f"{subcat_accuracy:.2f}%"
            })
        # 정확도 순으로 정렬
        subcat_stats.sort(key=lambda x: x['accuracy'], reverse=True)
        stats["by_sub_category"] = subcat_stats
    
    # 테이블/수식 포함 여부별 통계
    if 'has_table' in results_df.columns:
        table_stats = []
        for has_table in sorted(results_df['has_table'].unique()):
            table_df = results_df[results_df['has_table'] == has_table]
            table_total = len(table_df)
            table_correct = table_df[is_correct_col].sum()
            table_accuracy = (table_correct / table_total * 100) if table_total > 0 else 0.0
            table_stats.append({
                "has_table": has_table,
                "total": int(table_total),
                "correct": int(table_correct),
                "wrong": int(table_total - table_correct),
                "accuracy": round(table_accuracy, 2),
                "accuracy_percentage": f"{table_accuracy:.2f}%"
            })
        stats["by_has_table"] = table_stats
    
    if 'has_fomula' in results_df.columns:
        formula_stats = []
        for has_formula in sorted(results_df['has_fomula'].unique()):
            formula_df = results_df[results_df['has_fomula'] == has_formula]
            formula_total = len(formula_df)
            formula_correct = formula_df[is_correct_col].sum()
            formula_accuracy = (formula_correct / formula_total * 100) if formula_total > 0 else 0.0
            formula_stats.append({
                "has_formula": has_formula,
                "total": int(formula_total),
                "correct": int(formula_correct),
                "wrong": int(formula_total - formula_correct),
                "accuracy": round(formula_accuracy, 2),
                "accuracy_percentage": f"{formula_accuracy:.2f}%"
            })
        stats["by_has_formula"] = formula_stats
    
    return stats


def print_statistics(stats: Dict):
    """
    통계를 콘솔에 출력
    
    Args:
        stats: 통계 딕셔너리
    """
    print("\n" + "=" * 60)
    print(f"평가 결과 통계: {stats['model']}")
    print("=" * 60)
    print(f"전체 정확도: {stats['accuracy_percentage']} ({stats['correct_answers']}/{stats['total_questions']})")
    print(f"정답: {stats['correct_answers']}개")
    print(f"오답: {stats['wrong_answers']}개")
    
    if 'by_category' in stats:
        print("\n[카테고리별 정확도]")
        for cat in stats['by_category']:
            print(f"  {cat['category']}: {cat['accuracy_percentage']} ({cat['correct']}/{cat['total']})")
    
    if 'by_level' in stats:
        print("\n[레벨별 정확도]")
        for level in stats['by_level']:
            print(f"  {level['level']}: {level['accuracy_percentage']} ({level['correct']}/{level['total']})")
    
    if 'by_sub_category' in stats:
        print("\n[서브카테고리별 정확도 (상위 10개)]")
        for subcat in stats['by_sub_category'][:10]:
            print(f"  {subcat['sub_category']}: {subcat['accuracy_percentage']} ({subcat['correct']}/{subcat['total']})")
    
    if 'by_has_table' in stats:
        print("\n[테이블 포함 여부별 정확도]")
        for table in stats['by_has_table']:
            print(f"  테이블 {'포함' if table['has_table'] == 'Y' else '미포함'}: {table['accuracy_percentage']} ({table['correct']}/{table['total']})")
    
    if 'by_has_formula' in stats:
        print("\n[수식 포함 여부별 정확도]")
        for formula in stats['by_has_formula']:
            print(f"  수식 {'포함' if formula['has_formula'] == 'Y' else '미포함'}: {formula['accuracy_percentage']} ({formula['correct']}/{formula['total']})")
    
    print("=" * 60)


# =================================
# CSV 파일 처리 함수
# =================================
def process_csv(
    input_csv_path: str,
    output_csv_path: str,
    model_name: str
) -> None:
    """
    CSV 파일을 읽어서 정답 비교 및 통계 계산 수행
    
    Args:
        input_csv_path: 입력 CSV 파일 경로 (1_1_eval_knowledge.py에서 생성된 파일)
        output_csv_path: 출력 CSV 파일 경로 (정답 여부 컬럼이 추가된 파일)
        model_name: 모델 이름
    """
    # 입력 CSV 파일 읽기
    try:
        data = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return
    
    answer_col = f"answer"
    
    # 모델 답변 컬럼이 있는지 확인
    if answer_col not in data.columns:
        print(f"오류: '{answer_col}' 컬럼을 찾을 수 없습니다.")
        return
    
    # 정답 여부 확인 및 컬럼 추가
    is_correct_col = f"is_correct"
    data[is_correct_col] = data.apply(
        lambda row: check_answer(row[answer_col], row['gold']),
        axis=1
    )
    
    # 통계 계산
    stats = calculate_statistics(data, model_name)
    
    # 통계 출력
    print_statistics(stats)
    
    # 통계를 JSON 파일로 저장
    stats_output_path = output_csv_path.replace('.csv', '_stats.json')
    os.makedirs(os.path.dirname(stats_output_path) if os.path.dirname(stats_output_path) else ".", exist_ok=True)
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n통계 저장 완료: {stats_output_path}")
    
    # CSV 파일 저장 (UTF-8 BOM 추가하여 Excel에서 한글 깨짐 방지)
    os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
    data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"결과 저장 완료: {output_csv_path}")


# =================================
# 메인 실행 로직
# =================================
if __name__ == "__main__":
    try:
        # ==========================================
        # 1단계: 처리할 결과 파일 설정
        # 
        # 주의사항: 통계 계산 전에 결과 파일을 확인하세요.
        # - answer 컬럼이 비어있거나 형식이 잘못된 경우, answer_text 컬럼을 확인하여
        #   수작업으로 올바른 답변(A~E)을 answer 컬럼에 입력해주세요.
        # ==========================================
        # 결과 디렉토리
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results/1_fin_knowledge")
        
        # 처리할 파일 패턴 (예: 1_fin_knowledge_*_response.csv)
        # 또는 특정 파일 리스트 지정 가능
        input_pattern = os.path.join(results_dir, "1_fin_knowledge_*_response.csv")
        input_files = glob.glob(input_pattern)
        
        if not input_files:
            print(f"처리할 파일을 찾을 수 없습니다: {input_pattern}")
            exit(1)
        
        print("=" * 60)
        print("평가 결과 통계 계산 모드")
        print("=" * 60)
        print(f"처리할 파일: {len(input_files)}개")
        for f in input_files:
            print(f"  - {os.path.basename(f)}")
        print("=" * 60)
        
        # ==========================================
        # 2단계: 각 파일 처리
        # ==========================================
        for input_file in input_files:
            print(f"\n{'='*60}")
            print(f"처리 중: {os.path.basename(input_file)}")
            print(f"{'='*60}")
            
            # 파일명에서 모델명 추출
            # 예: 1_fin_knowledge_gpt-oss-20b_response.csv -> gpt-oss-20b
            filename = os.path.basename(input_file)
            # 파일명 패턴: {benchmark}_{model_name}_response.csv
            parts = filename.replace('_response.csv', '').split('_')
            # benchmark 이름 제거 (예: 1_fin_knowledge)
            if len(parts) >= 3:
                model_name = '_'.join(parts[3:])  # 모델명 부분만 추출
            else:
                # 패턴이 맞지 않으면 파일명에서 직접 추출 시도
                model_name = filename.replace('1_fin_knowledge_', '').replace('_response.csv', '')
            
            # 출력 파일 경로 (같은 디렉토리에 저장)
            output_file = input_file  # 같은 파일에 덮어쓰기 (정답 여부 컬럼 추가)
            
            # CSV 처리 및 통계 계산 실행
            process_csv(input_file, output_file, model_name)
            
            print(f"✓ {os.path.basename(input_file)} 처리 완료")
        
        print("\n" + "=" * 60)
        print("모든 파일 처리 완료!")
        print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()



