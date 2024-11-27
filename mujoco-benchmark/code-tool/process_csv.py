import os
import pandas as pd
import shutil

# 指定要处理的文件夹路径
folder_path = 'mujoco-benchmark/data-humanbench'  # 请将此路径替换为你需要操作的文件夹路径

# 遍历文件夹中的所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # 提取实验任务名称，假设格式为 "sac_<experiment_name>_alpha_<...>"
        try:
            experiment_name = filename.split('_')[1]  # 提取实验任务名称

            # 检查是否存在实验任务名称的文件夹，不存在则创建
            task_folder_path = os.path.join(folder_path, experiment_name)
            if not os.path.exists(task_folder_path):
                os.makedirs(task_folder_path)
                print(f"创建了实验任务文件夹: {experiment_name}")

            # 读取 CSV 文件
            df = pd.read_csv(file_path)

            # 筛选出列名中不包含 'MIN' 和 'MAX' 且满足条件的列
            columns_to_keep = [col for col in df.columns if 'MIN' not in col.upper() and 'MAX' not in col.upper()]
            df_filtered = df[columns_to_keep]

            # 重命名特定列名
            renamed_columns = {}
            for col in df_filtered.columns:
                if 'baseline_results/return' in col or 'results/return2' in col:
                    renamed_columns[col] = 'Baseline'
                elif 'correct_results/return' in col or 'results/return' in col:
                    renamed_columns[col] = 'Corrected'
                

            # 应用列名重命名
            df_filtered = df_filtered.rename(columns=renamed_columns)

            # 将处理后的 DataFrame 保存回原 CSV 文件，替换原文件
            df_filtered.to_csv(file_path, index=False)
            print(f"处理完成并替换了文件: {filename}")
            
            # 将 CSV 文件移动到相应的实验任务文件夹中
            new_file_path = os.path.join(task_folder_path, filename)
            shutil.move(file_path, new_file_path)
            print(f"已将文件 {filename} 移动到文件夹: {experiment_name}")
        
        except Exception as e:
            # 捕获任何错误并输出文件名和错误信息
            print(f"处理文件 {filename} 时出错: {e}")

print("所有 CSV 文件已处理完毕并分类。")
