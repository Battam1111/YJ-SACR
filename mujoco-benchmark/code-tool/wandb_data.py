import wandb
import pandas as pd

# 设置您的API对象
api = wandb.Api()

# 替换成您的项目名称和用户名（例如 "username/project_name"）
project_path = "battam/humanoid-bench"  # 请替换成您的项目路径

# 获取项目中所有的 runs
runs = api.runs(project_path)

# 循环遍历每个 run，获取所需数据，并保存为 CSV
for run in runs:
    # 动态生成列名
    baseline_column = f"{run.name}_baseline_results/return"
    corrected_column = f"{run.name}_correct_results/return"
    
    # 尝试获取 `Step`、动态列名的 `baseline_results/return` 和 `correct_results/return` 数据
    try:
        history = run.history(keys=["Step", baseline_column, corrected_column])
        print(f"{run.name} 包含的列：{list(history.columns)}")  # 打印列名以检查数据存在

        # 检查数据是否存在
        if not history.empty:
            # 重命名列以符合要求
            history = history.rename(columns={
                "Step": "Step",
                baseline_column: "Baseline",
                corrected_column: "Corrected"
            })

            # 创建 CSV 文件名，以 run 的 name 命名
            csv_filename = f"{run.name}.csv"
            
            # 将数据保存为 CSV
            history.to_csv(csv_filename, index=False)
            print(f"已保存 {csv_filename}")
        else:
            print(f"Run {run.name} 中没有所需的数据，跳过")
    except Exception as e:
        print(f"Run {run.name} 读取数据时发生错误：{e}")
