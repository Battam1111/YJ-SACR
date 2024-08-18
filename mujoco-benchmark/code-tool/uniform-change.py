import os
import pandas as pd

# 设定根文件夹路径和新列名
root_dir = 'mujoco-benchmark/data'
new_column_name = 'refineSampling'

# 遍历根目录下的所有子文件夹和文件
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            
            # 读取csv文件
            df = pd.read_csv(file_path)
            
            # 检查是否存在列名为 'uniform_v1'
            if 'uniform_v1' in df.columns:
                # 将列名 'uniform_v1' 修改为新的列名
                df.rename(columns={'uniform_v1': new_column_name}, inplace=True)
                
                # 保存修改后的csv文件
                df.to_csv(file_path, index=False)
                print(f'{file_path} 的列名已更新为 {new_column_name}')
            else:
                print(f'{file_path} 不包含列名 uniform_v1')

print("所有文件已处理完毕。")
