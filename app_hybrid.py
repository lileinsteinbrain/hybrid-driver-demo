# app_hybrid.py  — Hybrid demo (Qualifying only) + segment explain + optional CSV upload
import os, sys, time, json
import numpy as np
import pandas as pd
import streamlit as st
import requests, os
LIVE_URL = os.environ.get("LIVE_PUSH_URL", "http://localhost:8000/push")

# ==== render backend safe ====
os.environ.setdefault("MPLBACKEND", "Agg")

# ==== paths ====
ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT, "driver-fingerprint", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

RESULT_DIR   = os.path.join(ROOT, "driver-fingerprint", "results", "integration_Q5")
MODEL_PATH   = os.path.join(RESULT_DIR, "model.pth")
SUMMARY_JSON = os.path.join(RESULT_DIR, "summary_integration.json")

# ==== fastf1 cache: use /tmp on cloud ====
import fastf1
FASTF1_CACHE = os.path.join("/tmp", "fastf1_cache")
os.makedirs(FASTF1_CACHE, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE)

# ==== our utils / model ====
from utils_fp import (
    PolicyNet, nll_weighted,
    collect_driver_laps_resampled, build_dataset
)

# -------- feature helper for USER CSV (optional) --------
def make_features_from_csv(df_raw: pd.DataFrame, n_pts: int = 220):
    """
    期望列：
      - 必需：time_s, speed_kph, throttle(0-100), brake(0/1 或 0-100)
      - 可选：x, y（用于算 heading / a_lat），distance（没有就由速度积分）
    生成与你训练一致的特征：
      S: [phase, v/90, tanh(a_lat/6), tanh(a_long/5), tanh(mu_proxy/9.81)]
      A: [d_heading, d_brake, d_throttle]
      并在 phase∈[0,1] 的均匀网格上重采样到 220 点。
    """
    for col in ["time_s","speed_kph","throttle","brake"]:
        if col not in df_raw.columns:
            raise ValueError(f"CSV missing column: {col}")

    t = df_raw["time_s"].to_numpy().astype(float)
    v = df_raw["speed_kph"].to_numpy().astype(float) * 1000/3600.0
    thr = df_raw["throttle"].to_numpy().astype(float) / 100.0
    brk = df_raw["brake"].to_numpy().astype(float)
    brk = np.where(brk > 1.0, brk/100.0, brk)  # 兼容 0/100
    brk = np.clip(brk, 0.0, 1.0)

    # distance（若无则由速度积分近似）
    if "distance" in df_raw.columns:
        dist = df_raw["distance"].to_numpy().astype(float)
    else:
        dt = np.diff(t, prepend=t[0])
        dt[0] = dt[1] if len(dt)>1 else 0.01
        dist = np.cumsum(v * dt)

    # phase
    L = max(1e-6, (np.nanmax(dist) - np.nanmin(dist)))
    phase = (dist - np.nanmin(dist)) / L
    phase = np.clip(phase, 0, 1)

    # a_long
    a_long = np.gradient(v, t)

    # heading / a_lat（只有 x,y 时较准；否则置 0）
    if ("x" in df_raw.columns) and ("y" in df_raw.columns):
        x = df_raw["x"].to_numpy().astype(float)
        y = df_raw["y"].to_numpy().astype(float)
        vx = np.gradient(x, t); vy = np.gradient(y, t)
        heading = np.unwrap(np.arctan2(vy, vx))
        ds = np.clip(np.gradient(dist), 1e-3, None)
        kappa = np.gradient(heading)/ds
        a_lat = (v**2) * kappa
        d_head = np.gradient(heading, t)
    else:
        a_lat = np.zeros_like(v)
        d_head = np.zeros_like(v)

    d_thr = np.gradient(thr, t)
    d_brk = np.gradient(brk, t)
    mu_proxy = np.sqrt(a_lat**2 + a_long**2)

    # 重采样至固定网格
    def _resample(x_phase, y, grid):
        m = ~(np.isnan(x_phase) | np.isnan(y))
        if m.sum() < 8: return None
        x0, y0 = x_phase[m], y[m]
        o = np.argsort(x0); x1, y1 = x0[o], y0[o]
        x2, idx = np.unique(x1, return_index=True); y2 = y1[idx]
        return np.interp(grid, x2, y2)

    grid = np.linspace(0, 1, n_pts)
    feats = [phase, v/90.0, np.tanh(a_lat/6.0), np.tanh(a_long/5.0), np.tanh(mu_proxy/9.81)]
    acts  = [d_head, d_brk, d_thr]

    S_parts, A_parts = [], []
    for arr in feats:
        ri = _resample(phase, arr, grid)
        if ri is None: return None, None, None
        S_parts.append(ri)
    for arr in acts:
        ri = _resample(phase, arr, grid)
        if ri is None: return None, None, None
        A_parts.append(ri)

    S = np.stack(S_parts,1).astype(np.float32)
    A = np.stack(A_parts,1).astype(np.float32)
    M = pd.DataFrame({"driver":"USER","phase":grid})
    return [S], [A], [M]

# ==== Streamlit page ====
st.set_page_config(page_title="Hybrid Driver Demo", layout="wide")
st.sidebar.title("Hybrid Demo")
st.sidebar.caption("Real telemetry → similarity vs driver fingerprints (with Hybrid mix)")

# ---------------- Session selector ----------------
YEAR_OPTS = [2023, 2024]
EVENT_BY_YEAR = {
    2023: ["British Grand Prix","United States Grand Prix","Australian Grand Prix","Bahrain Grand Prix","Brazilian Grand Prix"],
    2024: ["British Grand Prix","United States Grand Prix","Australian Grand Prix","Bahrain Grand Prix","Brazilian Grand Prix"],
}

with st.sidebar.expander("Session"):
    year = st.selectbox("Year", YEAR_OPTS, index=0)
    event_name = st.selectbox("Event", EVENT_BY_YEAR[year], index=0)
    st.write("Session: **Qualifying (Q)**")

# read classes & best z from summary (fallback if not found)
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

# ---------------- Playback controls ----------------
with st.sidebar.expander("Playback"):
    play_speed = st.slider("Playback speed (steps/sec)", 1, 12, 4)
    auto_loop  = st.checkbox("Loop", True)

# ---------------- Hybrid controls ----------------
with st.sidebar.expander("Hybrid mix"):
    colA, colB = st.columns(2)
    with colA:
        drv_A = st.selectbox("Driver A", classes, index=0)
    with colB:
        drv_B = st.selectbox("Driver B", classes, index=1)
    alpha = st.slider("α (A weight)", 0.0, 1.0, 0.5, 0.05)

# ---------------- Data source + Language ----------------
ENABLE_USER_UPLOAD = True
user_csv = None
with st.sidebar.expander("Data source"):
    source = st.radio("Telemetry source", ["FastF1 (official)", "Upload CSV"], index=0)
    lang = st.radio("Language / 语言", ["English", "中文"], index=1, horizontal=True)
    if ENABLE_USER_UPLOAD and source == "Upload CSV":
        user_csv = st.file_uploader(
            "Upload lap CSV (cols: time_s, speed_kph, throttle, brake, optional x,y,distance)",
            type=["csv"]
        )

st.title("Hybrid Driver Similarity — Qualifying Telemetry vs Driver Fingerprints")
st.caption("Pick session & telemetry, then mix **Driver A/B** with α to generate a *Hybrid fingerprint* in real-time.")

# ---------------- Load session or user CSV ----------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_session(year: int, gp_name: str):
    sess = fastf1.get_session(year, gp_name, "Q")
    sess.load()
    return sess

if (user_csv is None) or (not ENABLE_USER_UPLOAD) or (source == "FastF1 (official)"):
    session = load_session(year, event_name)

# ---------------- Collect laps (resampled) ----------------
@st.cache_data(show_spinner=True, ttl=900)
def get_resampled_laps(_session, codes, n_pts=220, max_laps=3):
    packs = {}
    for code in codes:
        try:
            S_list, A_list, M_list = collect_driver_laps_resampled(
                _session, code, n_pts=n_pts, max_laps=max_laps
            )
        except Exception:
            S_list, A_list, M_list = [], [], []
        packs[code] = (S_list, A_list, M_list)
    return packs

packs = None
if (user_csv is None) or (source == "FastF1 (official)"):
    packs = get_resampled_laps(session, classes, n_pts=220, max_laps=3)

# Telemetry driver + which lap
c1, c2 = st.columns([2, 1])
if (user_csv is not None) and (source == "Upload CSV"):
    telem_options = ["USER"] + classes  # 允许“USER”作为源，也可切到官方某车手
else:
    telem_options = classes

with c1:
    telem_drv = st.selectbox("Pick a driver (telemetry source)", telem_options, index=0)

# 取 S/A/M
if telem_drv == "USER":
    try:
        df_u = pd.read_csv(user_csv)
        S_list, A_list, M_list = make_features_from_csv(df_u, n_pts=220)
        if (S_list is None) or (len(S_list)==0):
            st.error("CSV 解析失败：信号不足或缺字段。请检查列名/数据质量。")
            st.stop()
    except Exception as e:
        st.error(f"CSV 解析失败：{e}")
        st.stop()
else:
    S_list, A_list, M_list = packs.get(telem_drv, ([], [], []))

with c2:
    avail = len(S_list)
    lap_idx = st.number_input("Lap index (fastest-first)", min_value=0, max_value=max(0, avail-1), value=0, step=1)

if not S_list:
    st.error("No laps available for this driver/session. Try another driver or event.")
    st.stop()

# windows same as training
S_all, A_all, meta, D_lap, lap_sizes = build_dataset(
    [S_list[lap_idx]], [A_list[lap_idx]], [M_list[lap_idx]],
    use_window=True, windows=((0.15, 0.55), (0.60, 0.85))
)

# ---------------- Build model & load weights ----------------
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
sd, ad = S_all.shape[1], A_all.shape[1]

model = PolicyNet(sd=sd, ad=ad, n=len(classes), z_dim=best_z, hidden=96, use_embedding=True)
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)  # allow slight key mismatch
    except Exception as e:
        st.warning(f"Loading state_dict for PolicyNet: size mismatch (already handled); detail: {e}")
else:
    st.warning("model.pth not found — using randomly initialized weights (demo only).")
model.to(device).eval()

# normalize actions per-lap (same as training inference)
A_mu = A_all.mean(0, keepdims=True)
A_std = A_all.std(0, keepdims=True) + 1e-6
A_std_all = (A_all - A_mu) / A_std

to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
S_t = to_t(S_all)
A_t = to_t(A_std_all)

# time index state
if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0
nT = S_t.shape[0]

# ---------------- Core helpers ----------------
def get_z(driver_code: str):
    idx = drv_to_id[driver_code]
    with torch.no_grad():
        z = model.emb.weight[idx:idx+1]
    return z.to(device)

def _fwd_with_z_override_or_bypass(S_step: torch.Tensor, z: torch.Tensor):
    try:
        return model(S_step, None, z_override=z)  # 若你的 PolicyNet.forward 支持
    except TypeError:
        pass
    except Exception:
        pass
    if hasattr(model, "net"):
        x = torch.cat([S_step, z], dim=-1)
        out = model.net(x)
        m, lv = torch.chunk(out, 2, dim=-1)
        lv = torch.clamp(lv, -4.0, 2.0)
        return m, lv
    else:
        raise RuntimeError("Model does not expose z_override nor .net for bypass. Enable z_override in PolicyNet.forward.")

def step_nll_with_z(ti: int, z_override: torch.Tensor):
    with torch.no_grad():
        S_step = S_t[ti:ti+1]
        m, lv = _fwd_with_z_override_or_bypass(S_step, z_override)
        nll = nll_weighted(m, lv, A_t[ti:ti+1]).mean().item()
        return float(nll)

def step_nll_with_driver_id(ti: int, driver_code: str):
    d_id = drv_to_id[driver_code]
    with torch.no_grad():
        S_step = S_t[ti:ti+1]
        d_idx = torch.tensor([d_id], dtype=torch.long, device=device)
        m, lv = model(S_step, d_idx)
        nll = nll_weighted(m, lv, A_t[ti:ti+1]).mean().item()
        return float(nll)

def step_similarities(ti: int, drvA: str, drvB: str, alpha: float):
    nlls = [step_nll_with_driver_id(ti, name) for name in classes]
    nlls = np.array(nlls, dtype=np.float64)
    zA = get_z(drvA); zB = get_z(drvB)
    z_mix = alpha * zA + (1.0 - alpha) * zB
    nll_h = step_nll_with_z(ti, z_mix)
    sim = np.exp(-nlls); sim = sim / (sim.sum() + 1e-9)
    sim_h = float(np.exp(-nll_h))
    labels = classes + [f"Hybrid[{drvA}/{drvB}|α={alpha:.2f}]"]
    values = list(sim) + [sim_h]
    return labels, values

# ---------------- UI: playback controls ----------------
ctrl_col, bar_col = st.columns([1, 3])
with ctrl_col:
    if st.button("⏮ Reset"):
        st.session_state.t_idx = 0
    play = st.toggle("▶ Play", value=True)

labels_bar, values_bar = step_similarities(st.session_state.t_idx, drv_A, drv_B, alpha)

with bar_col:
    st.subheader("Live similarity (higher = closer to fingerprint)")
    df_bar = pd.DataFrame({"driver": labels_bar, "similarity": values_bar})
    st.bar_chart(df_bar.set_index("driver"))

def push_live_frame(t_idx, alpha, drv_A, drv_B, A_all_row, sim_labels, sim_values):
    sim = {k: float(v) for k,v in zip(sim_labels, sim_values)}
    feat = {
        "d_head": float(A_all_row[0]),
        "d_brake": float(A_all_row[1]),
        "d_thr": float(A_all_row[2]),
    }
    payload = {
        "t": int(t_idx),
        "alpha": float(alpha),
        "driverA": drv_A,
        "driverB": drv_B,
        "features": feat,
        "sim": sim,
    }
    try:
        requests.post(LIVE_URL, json=payload, timeout=0.2)
    except Exception:
        pass

# 计算好 labels_bar / values_bar 后，加：
push_live_frame(
    st.session_state.t_idx, alpha, drv_A, drv_B,
    A_all[st.session_state.t_idx], labels_bar, values_bar
)

# bottom curves
st.subheader(f"Lap playback — step {st.session_state.t_idx+1}/{nT}  |  Event: {event_name} {year} (Q)")
cols = st.columns(3)
labels_curve = ["d_heading", "d_brake", "d_throttle"]
for i, c in enumerate(cols):
    with c:
        st.line_chart(pd.DataFrame({labels_curve[i]: A_all[:st.session_state.t_idx+1, i]}))

# timeline
prog = st.slider("timeline", 0, nT-1, st.session_state.t_idx, key="timeline")
st.session_state.t_idx = prog

# autoplay
if play:
    time.sleep(1.0 / max(1, int(play_speed)))
    st.session_state.t_idx = (st.session_state.t_idx + 1) % nT if auto_loop else min(nT-1, st.session_state.t_idx+1)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# =======================
#  SEGMENT-BY-SEGMENT EXPLAIN
# =======================
st.markdown("---")
st.subheader("Explain this lap (segment-by-segment)")
n_segs = st.slider("How many segments", 4, 12, 6, help="按 phase 等分切段；后续可换成赛道/弯角切分")

def make_segments_by_phase(meta_df: pd.DataFrame, n_segs: int = 6):
    phases = meta_df["phase"].to_numpy() if isinstance(meta_df, pd.DataFrame) and ("phase" in meta_df.columns) else np.linspace(0, 1, len(meta_df))
    idx = np.arange(len(phases))
    edges = np.linspace(0.0, 1.0, n_segs + 1)
    segs = []
    for i in range(n_segs):
        lo, hi = edges[i], edges[i+1]
        m = (phases >= lo - 1e-9) & (phases <= hi + 1e-9)
        rows = idx[m]
        if len(rows) == 0: continue
        segs.append((rows.min(), rows.max()+1, f"S{i+1} {lo:.2f}-{hi:.2f}"))
    return segs

@st.cache_data(show_spinner=False)
def _softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.min()
    p = np.exp(-x)
    return p / (p.sum() + 1e-12)

def explain_one_segment(s_lo, s_hi, S_t, A_t, classes, drv_to_id, model, device):
    with torch.no_grad():
        nlls = []
        for name in classes:
            d_id = drv_to_id[name]
            d_vec = torch.full((s_hi - s_lo,), d_id, device=device, dtype=torch.long)
            m, lv = model(S_t[s_lo:s_hi], d_vec)
            nll = nll_weighted(m, lv, A_t[s_lo:s_hi]).mean().item()
            nlls.append(nll)
        probs = _softmax(nlls)
        sims = [(classes[i], float(probs[i]), float(nlls[i])) for i in range(len(classes))]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

# —— 维持你原来的均值/方差统计，不改口径 —— 
def action_summary(A_all, s_lo, s_hi, labels=("d_heading", "d_brake", "d_throttle")):
    seg = A_all[s_lo:s_hi]
    mean = seg.mean(axis=0); std  = seg.std(axis=0)
    return {labels[i]: (float(mean[i]), float(std[i])) for i in range(min(len(labels), seg.shape[1]))}

# —— 自然语言生成（新增）——
def explain_segments_text(df_explain: pd.DataFrame, lang: str = "中文"):
    lines = []
    for _, r in df_explain.iterrows():
        seg = str(r["segment"])
        d1, s1 = r["top1_driver"], r["top1_sim"]
        d2     = r["top2_driver"]
        conf   = r["confidence"]
        if lang == "English":
            lines.append(f"{seg}: most similar to **{d1}** (sim={s1:.3f}, Δ={conf:.3f} vs {d2}).")
        else:
            lines.append(f"{seg} 段：**最像 {d1}**（相似度 {s1:.3f}，领先 {d2} {conf:.3f}）。")
    return "\n".join(lines)

if play:
    st.info("⏸ 暂停播放以运行 Explain。")
else:
    if st.button("🧠 Explain current lap"):
        segs = make_segments_by_phase(meta, n_segs=n_segs) if (isinstance(meta, pd.DataFrame) and ("phase" in meta.columns)) \
               else [(int(c), int(d), f"S{i+1}") for i,(c,d) in enumerate(zip(np.linspace(0,S_t.shape[0],n_segs+1,dtype=int)[:-1],
                                                                              np.linspace(0,S_t.shape[0],n_segs+1,dtype=int)[1:])) if d>c]
        rows = []
        for (s_lo, s_hi, seg_name) in segs:
            sims = explain_one_segment(s_lo, s_hi, S_t, A_t, classes, drv_to_id, model, device)
            top1, top2 = sims[0], (sims[1] if len(sims) > 1 else ("-", 0.0, 0.0))
            conf = float(top1[1] - (top2[1] if isinstance(top2, tuple) else 0.0))
            summ = action_summary(A_all, s_lo, s_hi)
            rows.append({
                "segment": seg_name,
                "len": int(s_hi - s_lo),
                "top1_driver": top1[0],
                "top1_sim": round(float(top1[1]), 3),
                "top2_driver": top2[0] if isinstance(top2, tuple) else "-",
                "top2_sim": round(float(top2[1]), 3) if isinstance(top2, tuple) else 0.0,
                "confidence": round(conf, 3),
                "μ d_head": round(summ.get("d_heading", (0,0))[0], 3),
                "μ d_brake": round(summ.get("d_brake", (0,0))[0], 3),
                "μ d_thr": round(summ.get("d_throttle", (0,0))[0], 3),
            })
        df_explain = pd.DataFrame(rows, columns=[
            "segment","len","top1_driver","top1_sim","top2_driver","top2_sim","confidence",
            "μ d_head","μ d_brake","μ d_thr"
        ])
        st.dataframe(df_explain, use_container_width=True)
        st.caption("Top1 driver by segment")
        st.bar_chart(df_explain.set_index("segment")[["top1_sim"]])

# === 自然语言摘要（新增，强制显示在表格/柱状图后面）===
        st.markdown("#### Narrative / 文本说明")
        summary_text = explain_segments_text(df_explain, lang=lang)  # lang 来自侧栏 Language / 语言
        if lang == "English":
            st.info(summary_text or "No segment summary produced.")
        else:
            st.info(summary_text or "本圈未生成段落摘要。")

st.info("Similarity = exp(-NLL). We show class sims and also the **Hybrid(A,B,α)** sim computed via z_mix = α·z_A + (1-α)·z_B.")
st.caption("Quali only for now. Race & user-upload CSV will come next.")
