# app_hybrid.py
# ========== 一定要最先做：先建页，再做任何导入 ==========
import streamlit as st
st.set_page_config(page_title="Hybrid Driver Demo", layout="wide")

import os, sys, time, json
import numpy as np
import pandas as pd

# 云端画图更稳
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------- 1) 关键三方库导入：失败就把错误显示出来 ----------
try:
    import fastf1
except Exception as e:
    st.error(f"❌ 导入 fastf1 失败：{e}\n\n请确认 requirements.txt 已包含 fastf1。")
    st.stop()

try:
    import torch  # 仅导入；真正用时再检测设备
except Exception as e:
    st.error(
        f"❌ 导入 torch 失败：{e}\n\n"
        "请确认 requirements.txt 使用了 CPU 版 PyTorch 源：\n"
        "  torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu"
    )
    st.stop()

# ---------- 2) 让 Python 找到 utils_fp.py，失败也直接提示 ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT, "driver-fingerprint", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    from utils_fp import (
        PolicyNet, nll_weighted,
        collect_driver_laps_resampled, build_dataset
    )
except Exception as e:
    st.error(
        "❌ 无法导入 `utils_fp`。\n\n"
        f"搜索路径已加入：`{SCRIPTS_DIR}`\n"
        f"目录存在：{os.path.isdir(SCRIPTS_DIR)}\n"
        f"目录文件：{os.listdir(SCRIPTS_DIR) if os.path.isdir(SCRIPTS_DIR) else '（目录不存在）'}\n\n"
        f"错误：{e}"
    )
    st.stop()

# ---------- 3) FastF1 缓存 ----------
FASTF1_CACHE = os.path.join("/tmp", "fastf1_cache")
os.makedirs(FASTF1_CACHE, exist_ok=True)
try:
    fastf1.Cache.enable_cache(FASTF1_CACHE)
except Exception as e:
    st.warning(f"FastF1 缓存启用失败（已忽略）：{e}")

# ---------- 4) 结果路径 ----------
RESULT_DIR   = os.path.join(ROOT, "driver-fingerprint", "results", "integration_Q5")
MODEL_PATH   = os.path.join(RESULT_DIR, "model.pth")
SUMMARY_JSON = os.path.join(RESULT_DIR, "summary_integration.json")

# ---------- 侧边栏 ----------
st.sidebar.title("Hybrid Demo")
st.sidebar.caption("Real telemetry → similarity vs driver fingerprints")

YEAR_OPTS = [2023, 2024]
EVENT_BY_YEAR = {
    2023: ["British Grand Prix","United States Grand Prix","Australian Grand Prix","Bahrain Grand Prix","Brazilian Grand Prix"],
    2024: ["British Grand Prix","United States Grand Prix","Australian Grand Prix","Bahrain Grand Prix","Brazilian Grand Prix"],
}
with st.sidebar.expander("Session"):
    year = st.selectbox("Year", YEAR_OPTS, index=0)
    event_name = st.selectbox("Event", EVENT_BY_YEAR[year], index=0)
    st.write("Session: **Qualifying (Q)**")

@st.cache_resource(show_spinner=True)
def load_model_info(summary_json: str):
    classes = ["NOR", "RUS", "VER"]; z_dim = 16
    if os.path.exists(summary_json):
        try:
            with open(summary_json, "r") as f:
                info = json.load(f)
            classes = info.get("classes", classes)
            z_dim = int(info.get("best", {}).get("z_dim", z_dim))
        except Exception:
            pass
    return classes, z_dim

classes, best_z = load_model_info(SUMMARY_JSON)
drv_to_id = {d: i for i, d in enumerate(classes)}

with st.sidebar.expander("Drivers"):
    show_codes = st.multiselect("Compare drivers", options=classes, default=classes)

with st.sidebar.expander("Playback"):
    play_speed = st.slider("Playback speed (steps/sec)", 1, 12, 4)
    auto_loop  = st.checkbox("Loop", True)

st.title("Hybrid Driver Similarity — Qualifying Telemetry vs Driver Fingerprints")
st.caption("Pick drivers & session, play lap telemetry; see **live similarity** to each pro’s fingerprint.")

# ---------- 5) 加载会话 ----------
@st.cache_data(show_spinner=True, ttl=3600)
def load_session(year: int, gp_name: str):
    ses = fastf1.get_session(year, gp_name, "Q")
    ses.load()
    return ses

try:
    session = load_session(year, event_name)
except Exception as e:
    st.error(f"❌ FastF1 会话加载失败：{e}")
    st.stop()

# ---------- 6) 抽取圈数据（_session 规避 cache hash） ----------
@st.cache_data(show_spinner=True, ttl=600)
def get_resampled_laps(_session, codes, n_pts=220, max_laps=3):
    session = _session
    packs = {}
    for code in codes:
        try:
            S_list, A_list, M_list = collect_driver_laps_resampled(
                session, code, n_pts=n_pts, max_laps=max_laps
            )
        except Exception:
            S_list, A_list, M_list = [], [], []
        packs[code] = (S_list, A_list, M_list)
    return packs

packs = get_resampled_laps(session, show_codes, n_pts=220, max_laps=3)

# ---------- 7) 选择圈 ----------
c1, c2 = st.columns([1, 1])
with c1:
    drv = st.selectbox("Pick a driver", show_codes, index=0)
S_list, A_list, M_list = packs.get(drv, ([], [], []))
with c2:
    avail = len(S_list)
    lap_idx = st.number_input("Lap index (fastest first)", 0, max(0, avail-1), 0, 1)

if not S_list:
    st.error("该会话/车手没有可用的圈。请换一位车手或赛事。")
    st.stop()

# ---------- 8) 与训练一致：两个窗口 ----------
try:
    S_all, A_all, meta, D_lap, lap_sizes = build_dataset(
        [S_list[lap_idx]], [A_list[lap_idx]], [M_list[lap_idx]],
        use_window=True, windows=((0.15, 0.55), (0.60, 0.85))
    )
except Exception as e:
    st.error(f"❌ 构建窗口化数据失败：{e}")
    st.stop()

# ---------- 9) 构网并载权重 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
sd, ad = S_all.shape[1], A_all.shape[1]
model = PolicyNet(sd=sd, ad=ad, n=len(classes), z_dim=best_z, hidden=96, use_embedding=True)

if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
    except Exception as e:
        st.warning(f"⚠️ 加载权重时有兼容性告警（已忽略）：{e}")
else:
    st.warning("⚠️ 未找到 model.pth，使用随机初始化权重（仅演示）。")

model.to(device).eval()

# 动作 z-score
A_mu = A_all.mean(0, keepdims=True)
A_std = A_all.std(0, keepdims=True) + 1e-6
A_std_all = (A_all - A_mu) / A_std

to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
S_t = to_t(S_all)
A_t = to_t(A_std_all)

# ---------- 10) 播放与可视化 ----------
if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0
nT = S_t.shape[0]

ctrl_col, bar_col = st.columns([1, 3])
with ctrl_col:
    if st.button("⏮ Reset"):
        st.session_state.t_idx = 0
    play = st.toggle("▶ Play", value=True)

def step_similarity(ti: int):
    with torch.no_grad():
        nlls = []
        for name in classes:
            d_id = drv_to_id[name]
            d_vec = torch.full((1,), d_id, device=device, dtype=torch.long)
            m, lv = model(S_t[ti:ti+1], d_vec)
            nll = nll_weighted(m, lv, A_t[ti:ti+1]).mean().item()
            nlls.append(nll)
        nlls = np.array(nlls, dtype=np.float64)
        sim = np.exp(-nlls); sim = sim / (sim.sum() + 1e-9)
        return sim, nlls

sim, _ = step_similarity(st.session_state.t_idx)
with bar_col:
    st.subheader("Live similarity (higher = closer to driver’s fingerprint)")
    st.bar_chart(pd.DataFrame({"driver": classes, "similarity": sim}).set_index("driver"))

st.subheader(f"Lap playback — step {st.session_state.t_idx+1}/{nT}  |  Event: {event_name} {year} (Q)")
cols = st.columns(3); labels = ["d_heading", "d_brake", "d_throttle"]
for i, c in enumerate(cols):
    with c:
        st.line_chart(pd.DataFrame({labels[i]: A_all[:st.session_state.t_idx+1, i]}))

prog = st.slider("timeline", 0, max(0, nT-1), st.session_state.t_idx, key="timeline")
st.session_state.t_idx = prog

if play and nT > 0:
    time.sleep(1.0 / max(1, int(play_speed)))
    st.session_state.t_idx = (st.session_state.t_idx + 1) % nT if auto_loop else min(nT-1, st.session_state.t_idx+1)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

st.info("Similarity uses exp(-NLL). Lower NLL ⇒ higher similarity.")
st.caption("Race 暂未开放；验证稳定后再放开；后续会加入上传 F1 game CSV 的入口。")
