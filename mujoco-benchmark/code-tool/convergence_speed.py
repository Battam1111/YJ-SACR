import pandas as pd

# Load the CSV data into a DataFrame
data = pd.read_csv('mujoco-benchmark/data/HumanoidStandup-v4/AF0.2/TestRew/Seed0_20240730-165956.csv')  # Replace 'your_file_path.csv' with your actual file path

# Calculate the difference between each consecutive epoch's rewards (i.e., the gradient)
data['deterministic_diff'] = data['deterministic'].diff().fillna(0)
data['uniform_v1_diff'] = data['uniform_v1'].diff().fillna(0)

# Calculate the convergence speed as the average of these differences
convergence_speed_deterministic = data['deterministic_diff'].mean()
convergence_speed_uniform_v1 = data['uniform_v1_diff'].mean()

print(f"Convergence Speed for Deterministic: {convergence_speed_deterministic}")
print(f"Convergence Speed for Uniform v1: {convergence_speed_uniform_v1}")
