import pandas as pd
import numpy as np
import os

def clean_data(input_path, output_path):
    print(f"开始清洗数据: {input_path}")
    
    # 1. 加载数据 (跳过前两行标题)
    df = pd.read_excel(input_path, skiprows=2, header=None)
    df.columns = ['Stkcd', 'Trddt', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dretwd', 'Dretnd', 'PreClosePrice', 'ChangeRatio', 'LimitDown', 'LimitUp', 'LimitStatus']
    
    initial_count = len(df)
    
    # 2. 基础清洗：处理缺失值和类型转换
    df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
    
    # 显式转换数值列，处理可能的非数值字符
    numeric_cols = ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'ChangeRatio', 'PreClosePrice', 'LimitUp', 'LimitDown']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=['Trddt', 'Stkcd', 'Clsprc', 'PreClosePrice', 'LimitUp'])
    
    # 3. 剔除停牌数据
    df = df[~((df['Opnprc'] == df['Clsprc']) & (df['Hiprc'] == df['Loprc']) & (df['ChangeRatio'] == 0))]
    after_halt_count = len(df)
    
    # 4. 剔除 ST 股票 (涨停限制约为 5%)
    df['Limit_Ratio'] = (df['LimitUp'] / df['PreClosePrice'] - 1).round(4)
    df = df[~((df['Limit_Ratio'] > 0.048) & (df['Limit_Ratio'] < 0.052))]
    after_st_count = len(df)
    
    # 5. 剔除新股 (上市不满一年的新股)
    df = df.sort_values(['Stkcd', 'Trddt'])
    df['First_Date'] = df.groupby('Stkcd')['Trddt'].transform('min')
    df['Days_Since_Listing'] = (df['Trddt'] - df['First_Date']).dt.days
    df = df[df['Days_Since_Listing'] > 250]
    after_new_count = len(df)
    
    # 6. 剔除异常值
    df = df[df['ChangeRatio'].abs() <= 0.21]
    final_count = len(df)
    
    # 保存清洗后的数据
    df.to_csv(output_path, index=False)
    
    print(f"清洗完成:")
    print(f"- 原始样本数: {initial_count}")
    print(f"- 剔除停牌后: {after_halt_count}")
    print(f"- 剔除 ST 后: {after_st_count}")
    print(f"- 剔除新股后: {after_new_count}")
    print(f"- 最终样本数: {final_count}")
    
    with open("/mnt/desktop/swufe_mcm/cleaning_log.txt", "w") as f:
        f.write(f"数据清洗报告\n")
        f.write(f"====================\n")
        f.write(f"原始样本数: {initial_count}\n")
        f.write(f"剔除停牌数: {initial_count - after_halt_count}\n")
        f.write(f"剔除 ST 股数: {after_halt_count - after_st_count}\n")
        f.write(f"剔除新股样本数: {after_st_count - after_new_count}\n")
        f.write(f"剔除异常值数: {after_new_count - final_count}\n")
        f.write(f"最终有效样本数: {final_count}\n")

if __name__ == "__main__":
    input_file = "/mnt/desktop/swufe_mcm/数据/TRD_Dalyr.xlsx"
    output_file = "/mnt/desktop/swufe_mcm/数据/TRD_Dalyr_Cleaned.csv"
    clean_data(input_file, output_file)
