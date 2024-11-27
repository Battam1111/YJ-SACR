conda创建一个3.10版本的python环境

然后再
pip install -r mujoco-benchmark/requirements.txt

pip install torch==2.2.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

然后替换若干文件（路径信息自行查看）
/anaconda3/envs/sacR/lib/python3.10/site-packages/tianshou/policy/modelfree/sac.py
/anaconda3/envs/sacR/lib/python3.10/site-packages/tianshou/data/collector.py
/anaconda3/envs/sacR/lib/python3.10/site-packages/torch/distributions/normal.py
/anaconda3/envs/sacR/lib/python3.10/site-packages/torch/distributions/independent.py
/anaconda3/envs/sacR/lib/python3.10/site-packages/tianshou/policy/base.py
/anaconda3/envs/sacR/lib/python3.10/site-packages/tianshou/trainer/base.py
/anaconda3/envs/sacR/lib/python3.10/site-packages/tianshou/trainer/utils.py

运行代码文件主要两个（自行调整相关参数）：
python mujoco-benchmark/mujoco_sac_refineTest_v1-ExpZhang.py
python mujoco-benchmark/mujoco_sac_refineTest_v1.py

任务/环境信息的网站：https://gymnasium.farama.org/environments/mujoco/
