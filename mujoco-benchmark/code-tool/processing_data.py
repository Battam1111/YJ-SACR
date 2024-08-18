import os
import pandas as pd

def remove_fourth_column_from_csv_files(directory):
    """删除指定文件夹中所有CSV文件的第四列数据"""
    # 遍历文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            # 读取CSV文件
            df = pd.read_csv(filepath, header=None)
            # 删除第四列（索引为3）
            if df.shape[1] > 3:
                df.drop(df.columns[3], axis=1, inplace=True)
                # 保存修改后的CSV文件
                df.to_csv(filepath, header=False, index=False)
                print(f"已删除文件 {filename} 的第四列数据")

# 示例用法
directory_path = 'Data-Buffer/data_0/Humanoid-v4/AF0.2/TestRew'  # 替换为你的文件夹路径
remove_fourth_column_from_csv_files(directory_path)
