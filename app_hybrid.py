# app_hybrid.py
import os
import time
import json
import sys
import numpy as np
import pandas as pd
import streamlit as st

# --- 禁用交互式后端，云端画图更稳 ---
os.environ.setdefault("MPLBACKEND", "Agg")

# --- 让 Python 能找到 driver-fingerprint/scripts/utils_fp.py ---
ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT, "driver-fingerprint", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# --- FastF1 缓存用 /tmp，Cloud 可写；本地也没问题 ---
import fastf1
FASTF1_CACHE = os.path.join("/tmp", "fastf1_cache")
os.makedirs(FASTF1_CACHE, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE)

# --- 我们自己的工具 ---
from utils_fp import (
    PolicyNet, nll_weighted,
    collect_driver_laps_resampled, build_dataset
)

# ---------- 固定路径 ----------
RESULT_DIR   = os.path.join(ROOT, "driver-fingerprint", "results", "integration_Q5")
MODEL_PATH   = os.path.join(RESULT_DIR, "model.pth")
SUMMARY_JSON = os.path.join(RESULT_DIR, "summary_integration.json")

# ---------- Streamlit 页面 ----------
st.set_page_config(page_title="Hybrid Driver Demo", layout="wide")
st.sidebar.title("Hybrid Demo")
st.sidebar.caption("Real telemetry → similarity vs driver fingerprints")

# 年份/赛事（只开 Q）
YEAR_OPTS = [2023, 2024]
EVENT_BY_YEAR = {
    2023: [
        "British Grand Prix",
        "United States Grand Prix",
        "Australian Grand Prix",
        "Bahrain Grand Prix",
        "Brazilian Grand Prix",
    ],
    2024: [
        "British Grand Prix",
        "United States Grand Prix",
        "Australian Grand Prix",
        "Bahrain Grand Prix",
        "Brazilian Grand Prix",
    ],
}

with st.sidebar.expander("Session"):
    year = st.selectbox("Year", YEAR_OPTS, index=0)
    event_name = st.selectbox("Event", EVENT_BY_YEAR[year], index=0)
    st.write("Session: **Qualifying (Q)**")  # Race 以后再放开

# 读取 classes / z_dim（来自训练摘要）
@st.cache_resource(show_spinner=True)
def load_model_info(summary_json: str):
    classes = ["NOR", "RUS", "VER"]
    z_dim = 16
    if os.path.exists(summary_json):
        try:
            info = json.load(open(summary_json, "r"))
            classes = info.get("classes", classes)
            z_dim = int(info.get("best", {}).get("z_dim", 16))
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

# ---------- 加载会话 ----------
@st.cache_data(show_spinner=True, ttl=3600)
def load_session(year: int, gp_name: str):
    sess = fastf1.get_session(year, gp_name, "Q")  # 只用 Q
    sess.load()
    return sess

session = load_session(year, event_name)

# ---------- 提取/重采样若干圈（注意：_session，避免 cache 对象哈希报错） ----------
@st.cache_data(show_spinner=True, ttl=600)
def get_resampled_laps(_session, codes, n_pts=220, max_laps=3):
    session = _session  # 只为避免 Streamlit 对 Session 做 hash
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

# 选择 driver & 圈（选最快的前几圈里某一圈）
c1, c2 = st.columns([1, 1])
with c1:
    drv = st.selectbox("Pick a driver", show_codes, index=0)
S_list, A_list, M_list = packs.get(drv, ([], [], []))
with c2:
    avail = len(S_list)
    lap_idx = st.number_input("Lap index (fastest first)", min_value=0, max_value=max(0, avail-1), value=0, step=1)

if not S_list:
    st.error("No laps available for this driver/session. Try another driver or event.")
    st.stop()

# 与训练一致：切两个窗口 (0.15–0.55) & (0.60–0.85)
S_all, A_all, meta, D_lap, lap_sizes = build_dataset(
    [S_list[lap_idx]], [A_list[lap_idx]], [M_list[lap_idx]],
    use_window=True, windows=((0.15, 0.55), (0.60, 0.85))
)

# ---------- 构图并载权重 ----------
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
sd, ad = S_all.shape[1], A_all.shape[1]

model = PolicyNet(sd=sd, ad=ad, n=len(classes), z_dim=best_z, hidden=96, use_embedding=True)
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)  # strict=False 防止 key 轻微不一致
    except Exception as e:
        st.warning(f"Weight load mismatch (ok for demo): {e}")
else:
    st.warning("model.pth not found — using randomly initialized weights (demo only).")
model.to(device).eval()

# 动作 z-score（和训练一致）
A_mu = A_all.mean(0, keepdims=True)
A_std = A_all.std(0, keepdims=True) + 1e-6
A_std_all = (A_all - A_mu) / A_std

to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
S_t = to_t(S_all)
A_t = to_t(A_std_all)

# 播放控件
if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0
nT = S_t.shape[0]

ctrl_col, bar_col = st.columns([1, 3])
with ctrl_col:
    if st.button("⏮ Reset"):
        st.session_state.t_idx = 0
    play = st.toggle("▶ Play", value=True)

# 一步 similarity（softmax(-NLL)）
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
        sim = np.exp(-nlls)
        sim = sim / (sim.sum() + 1e-9)
        return sim, nlls

sim, nlls = step_similarity(st.session_state.t_idx)

with bar_col:
    st.subheader("Live similarity (higher = closer to driver’s fingerprint)")
    df_bar = pd.DataFrame({"driver": classes, "similarity": sim})
    st.bar_chart(df_bar.set_index("driver"))

# 底部实时动作曲线
st.subheader(f"Lap playback — step {st.session_state.t_idx+1}/{nT}  |  Event: {event_name} {year} (Q)")
cols = st.columns(3)
labels = ["d_heading", "d_brake", "d_throttle"]
for i, c in enumerate(cols):
    with c:
        st.line_chart(pd.DataFrame({labels[i]: A_all[:st.session_state.t_idx+1, i]}))

# 时间轴
prog = st.slider("timeline", 0, nT-1, st.session_state.t_idx, key="timeline")
st.session_state.t_idx = prog

# 自动播放
if play:
    time.sleep(1.0 / max(1, int(play_speed)))
    st.session_state.t_idx = (st.session_state.t_idx + 1) % nT if auto_loop else min(nT-1, st.session_state.t_idx+1)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

st.info("Similarity uses exp(-NLL). Lower NLL ⇒ higher similarity.")
st.caption("Race 暂未开放；等我们验证完 Race 的稳定性再放开。Upload F1 game CSV 的入口也会在后续版本加入。")
