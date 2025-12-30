import re
import pandas as pd
import random
from typing import List, Dict

# -------------------------------
# 항 기호 정규식 (①~⑳)
PAT_HANG = re.compile(r'^[\u2460-\u2473]')

# -------------------------------
# 법령 제n조 인식: <<법령 제n조>>
PAT_LAW_PREFIX = re.compile(r'<<(.+?)>>')

# -------------------------------
# 항 단위 분리 함수 (수정됨)
def split_units_with_law_prefix(text: str) -> List[str]:
    results: List[str] = []

    # 전체 법령-조 블록 단위로 분리
    law_blocks = PAT_LAW_PREFIX.split(text)
    # 홀수 인덱스 = 법령, 짝수 인덱스 = 본문
    i = 1
    while i < len(law_blocks):
        law_name = law_blocks[i].strip()
        body = law_blocks[i+1] if i+1 < len(law_blocks) else ''
        body_lines = body.splitlines()

        current_hang_prefix = None  # 현재 항 접두사
        current_body_lines = []

        for line in body_lines:
            stripped = line.strip()
            if not stripped:
                continue  # 빈 줄 무시
            
            hang_match = PAT_HANG.match(stripped)
            if hang_match:
                # 이전 항이 있으면 flush
                if current_hang_prefix is not None:
                    content = '\n'.join(current_body_lines).strip()
                    if content:
                        results.append(f"{current_hang_prefix} {content}")
                
                # 새 항 시작
                hang_symbol = hang_match.group(0)
                after_text = stripped[len(hang_symbol):].strip()
                current_hang_prefix = f"<<{law_name}>> {hang_symbol}"
                current_body_lines = [after_text] if after_text else []
            else:
                # 항 기호가 아닌 경우 (호, 목 등)
                if current_hang_prefix is not None:
                    # 현재 항에 추가
                    current_body_lines.append(stripped)
                else:
                    # 항이 시작되기 전의 내용 (전문 등)
                    current_body_lines.append(stripped)

        # 마지막 항 또는 남은 내용 flush
        if current_hang_prefix is not None:
            content = '\n'.join(current_body_lines).strip()
            if content:
                results.append(f"{current_hang_prefix} {content}")
        elif current_body_lines:
            content = '\n'.join(current_body_lines).strip()
            if content:
                results.append(f"<<{law_name}>> {content}")
        
        i += 2

    return results

# -------------------------------
# Shuffle strategies
def position_based_shuffle(split_units: List[Dict], strategy: str = 'balance', cluster_position: str = 'front') -> List[Dict]:
    if not split_units:
        return []

    related_units = [u for u in split_units if u.get('related')]
    nonrelated_units = [u for u in split_units if not u.get('related')]

    if strategy == 'random':
        merged = split_units.copy()
        random.shuffle(merged)
        return merged

    if strategy == 'cluster':
        random.shuffle(related_units)
        random.shuffle(nonrelated_units)
        if cluster_position == 'front':
            return related_units + nonrelated_units
        elif cluster_position == 'middle':
            mid = len(nonrelated_units) // 2
            return nonrelated_units[:mid] + related_units + nonrelated_units[mid:]
        elif cluster_position in ('end','back'):
            return nonrelated_units + related_units
        else:
            raise ValueError("cluster_position must be 'front','middle','end'")

    if strategy == 'balance':
        random.shuffle(related_units)
        random.shuffle(nonrelated_units)
        total_len = len(related_units) + len(nonrelated_units)
        result: List[Dict] = []
        r_idx = u_idx = 0
        while len(result) < total_len:
            remain_r = len(related_units) - r_idx
            remain_u = len(nonrelated_units) - u_idx
            if remain_r <= 0:
                result.append(nonrelated_units[u_idx]); u_idx += 1; continue
            if remain_u <= 0:
                result.append(related_units[r_idx]); r_idx += 1; continue
            p_r = remain_r / (remain_r + remain_u)
            if random.random() < p_r:
                result.append(related_units[r_idx]); r_idx += 1
            else:
                result.append(nonrelated_units[u_idx]); u_idx += 1
        return result

    raise ValueError("strategy must be 'balance','cluster','random'")

# -------------------------------
# units -> text
def units_to_text(units: List[Dict]) -> str:
    return "\n\n".join([u.get('text','') for u in units])

# -------------------------------
# CSV 처리 최종
def process_csv_final(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv, dtype=str).fillna('')
    all_split_units = []

    for idx, row in df.iterrows():
        units: List[Dict] = []

        context_text = row.get('context','') or ''
        if context_text.strip():
            parsed_units = split_units_with_law_prefix(context_text)
            units.extend([{'text': t, 'related': True} for t in parsed_units])

        expand_text = row.get('expand_context','') or ''
        if expand_text.strip():
            parsed_units = split_units_with_law_prefix(expand_text)
            units.extend([{'text': t, 'related': False} for t in parsed_units])

        all_split_units.append(units)

    df['split_units'] = all_split_units

    # Shuffle variants
    df['shuffled_balance'] = df['split_units'].apply(lambda x: position_based_shuffle(x,'balance'))
    df['shuffled_cluster_front'] = df['split_units'].apply(lambda x: position_based_shuffle(x,'cluster','front'))
    df['shuffled_cluster_middle'] = df['split_units'].apply(lambda x: position_based_shuffle(x,'cluster','middle'))
    df['shuffled_cluster_end'] = df['split_units'].apply(lambda x: position_based_shuffle(x,'cluster','end'))
    df['shuffled_random'] = df['split_units'].apply(lambda x: position_based_shuffle(x,'random'))

    # Text-only columns
    df['shuffled_balance_text'] = df['shuffled_balance'].apply(units_to_text)
    df['shuffled_cluster_front_text'] = df['shuffled_cluster_front'].apply(units_to_text)
    df['shuffled_cluster_middle_text'] = df['shuffled_cluster_middle'].apply(units_to_text)
    df['shuffled_cluster_end_text'] = df['shuffled_cluster_end'].apply(units_to_text)
    df['shuffled_random_text'] = df['shuffled_random'].apply(units_to_text)

    df.to_csv(output_csv,index=False)
    return df

# # -------------------------------
# # 실행 예시
# if __name__ == "__main__":
#     sample_context = """<<자본시장법 제9조>> 

# ① 투자자는 다음과 같이 구분된다. 

# 1. 전문투자자 

# 2. 일반투자자 

# ② 다음 각 호의 행위는 금지된다. 가. 부정거래행위 나. 시세조종행위"""

#     test_df = pd.DataFrame([{'context': sample_context,'expand_context': ''}])
#     test_df.to_csv("test_input.csv", index=False)
#     out = process_csv_final("test_input.csv","test_output.csv")

#     print("=== 분리 결과 ===")
#     for i, unit in enumerate(out['split_units'][0], 1):
#         print(f"\n[Unit {i}]")
#         print(unit['text'])
#         print("-" * 50)
# ```

# ## 주요 변경사항

# 1. **항을 만나면 즉시 추가하지 않고 누적 시작**
# 2. **다음 항을 만날 때 이전 항을 flush** 
# 3. **호/목은 항 기호가 아니므로 자동으로 현재 항에 포함**

# ## 결과 예시
# ```
# [Unit 1]
# <<자본시장법 제9조>> ① 투자자는 다음과 같이 구분된다.
# 1. 전문투자자
# 2. 일반투자자

# [Unit 2]
# <<자본시장법 제9조>> ② 다음 각 호의 행위는 금지된다. 가. 부정거래행위 나. 시세조종행위



df_final = process_csv_final(
    "./data/context_preprocessing/(FN)12_금융위_금감원_금융규제_법령해석포털 회신사례_70건.csv",
    "./output/context_preprocessing/law_data_final_v8.csv"
)

# 결과 컬럼 확인
row = df_final.iloc[0]

print("=== split_units ===")
for u in row['split_units']:
    print(u)

print("\n=== shuffled_balance ===")
for u in row['shuffled_balance']:
    print(u['text'])

print("\n=== shuffled_cluster_front ===")
for u in row['shuffled_cluster_front']:
    print(u['text'])

print("\n=== shuffled_random ===")
for u in row['shuffled_random']:
    print(u['text'])
