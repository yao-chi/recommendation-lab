"""全局配置：路径、随机种子、实验常量。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "ml-1m"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
REPORT_DIR = ROOT / "reports"

for _p in (FIG_DIR, TAB_DIR, REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20261106

# 召回评测
K_LIST = (10, 20, 50)
MAIN_K = 20

# 排序 / CTR 评测
N_NEG_EVAL = 99          # 每个用户采样负例数（100 候选里 1 正 99 负）
CTR_USERS = 3000         # 参与 CTR 建模与评估的用户数（控制运行时间）
RANK_K = 10

# 冷启动分桶（按训练集交互数）
COLD_BINS = (0, 5, 20, 50)

TORCH_EPOCHS = 12
TORCH_BATCH = 4096
EMB_DIM = 64