# driver-fingerprint/scripts/export_persona.py
import os, json, numpy as np
from sklearn.decomposition import PCA
import torch
import sys

# 让 Python 找到 utils_fp.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "driver-fingerprint", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from utils_fp import PolicyNet, nll_weighted  # 只为取 embedding 结构

# 配置（按你现有路径）
RESULT_DIR   = os.path.join(ROOT, "driver-fingerprint", "results", "integration_Q5")
DATA_PATH    = os.path.join(ROOT, "driver-fingerprint", "integration", "data_integration.npz")
SUMMARY_JSON = os.path.join(RESULT_DIR, "summary_integration.json")
MODEL_PATH   = os.path.join(RESULT_DIR, "model.pth")
OUT_JSON     = os.path.join(RESULT_DIR, "persona_summary.json")

def rms(x): return float(np.sqrt(np.mean(x**2)))
def var(x): return float(np.var(x))

def main():
    # 1) 读数据/类名
    data = np.load(DATA_PATH, allow_pickle=True)
    S_all   = data["S_all"]
    A_all   = data["A_all"]   # [:, :, 0:3] ~ [d_head, d_brake, d_thr]
    D_lap   = data["D_lap"]   # 可能是字符串或 id
    lapsz   = data["lap_sizes"]

    # 类名顺序从 summary_integration.json 取（与训练一致）
    classes = ["NOR","RUS","VER"]
    z_dim = 16
    try:
        info = json.load(open(SUMMARY_JSON, "r"))
        classes = info.get("classes", classes)
        z_dim = int(info.get("best", {}).get("z_dim", 16))
    except Exception:
        pass

    # 标签→id
    if D_lap.dtype.kind in {"U","S","O"}:
        names = np.asarray(D_lap).astype(str)
        uniq  = sorted(np.unique(names))
        name2id = {d:i for i,d in enumerate(uniq)}
        D_id = np.array([name2id[d] for d in names], dtype=np.int64)
        # 重映射成 classes 顺序
        remap = {name2id[d]: i for i,d in enumerate(classes)}
        D_lap_id = np.array([remap[idx] for idx in D_id], dtype=np.int64)
    else:
        D_lap_id = np.asarray(D_lap, dtype=np.int64)

    # 2) 取每个 driver 的 A_all（拼回整圈）
    seg = np.cumsum(np.r_[0, lapsz])
    # 展开成逐时刻数组
    A_flat = []
    D_flat = []
    for k in range(len(lapsz)):
        rng = np.arange(seg[k], seg[k+1])
        A_flat.append(A_all[rng])
        D_flat.append(np.full(len(rng), D_lap_id[k], dtype=np.int64))
    A_flat = np.concatenate(A_flat, axis=0)  # [Tsum, 3]
    D_flat = np.concatenate(D_flat, axis=0)  # [Tsum]

    # 3) 从 model.pth 里取 embedding（z_d）
    sd, ad = S_all.shape[1], A_all.shape[1]
    device = "cpu"
    model = PolicyNet(sd=sd, ad=ad, n=len(classes), z_dim=z_dim, hidden=96, use_embedding=True).to(device)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        z_all = model.emb.weight.detach().cpu().numpy()  # [K, z_dim]

    # 4) 指标（E/T/S）按 driver 聚合
    feats = {i: {"thr":[], "brk":[], "hd":[]} for i in range(len(classes))}
    for i in range(len(classes)):
        idx = (D_flat == i)
        if idx.sum() == 0:  # 容错
            feats[i]["thr"] = [np.zeros(1)]
            feats[i]["brk"] = [np.zeros(1)]
            feats[i]["hd"]  = [np.zeros(1)]
        else:
            feats[i]["thr"] = [A_flat[idx][:,2]]
            feats[i]["brk"] = [A_flat[idx][:,1]]
            feats[i]["hd"]  = [A_flat[idx][:,0]]

    raw = {}
    for i, name in enumerate(classes):
        thr = np.concatenate(feats[i]["thr"]); brk = np.concatenate(feats[i]["brk"]); hd = np.concatenate(feats[i]["hd"])
        E = rms(thr) + 0.7*rms(brk)                  # Energy
        T = rms(hd)  + 0.3*rms(np.diff(hd))          # Tension
        S = 1.0 / (1.0 + var(thr) + var(brk))        # Smoothness
        raw[name] = {"E":E, "T":T, "S":S}

    # 5) embedding 做 PCA→ (p1,p2)
    pca = PCA(n_components=2)
    P = pca.fit_transform(z_all)  # [K,2]
    for i, name in enumerate(classes):
        raw[name]["p1"] = float(P[i,0])
        raw[name]["p2"] = float(P[i,1])

    # 6) 全体 min-max 归一到 [0,1]
    keys = ["E","T","S","p1","p2"]
    norm = {}
    for k in keys:
        v = np.array([raw[n][k] for n in classes], dtype=float)
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-9:
            for n in classes:
                norm.setdefault(n, {})[k] = 0.5
        else:
            for j, n in enumerate(classes):
                norm.setdefault(n, {})[k] = float((raw[n][k] - lo)/(hi-lo))

    # 7) 公开的“可追溯”音乐映射（写死公式）
    def to_params(n):
        E,T,S,p1,p2 = norm[n]["E"], norm[n]["T"], norm[n]["S"], norm[n]["p1"], norm[n]["p2"]
        bpm   = int(100 + 60*E)               # 100..160
        swing = 0.02 + 0.08*T                 # 0.02..0.10
        hat   = 0.2 + 0.7*(0.6*E + 0.4*T)
        kick  = 0.3 + 0.7*E
        lead  = 0.3 + 0.7*S
        bass  = 0.4 + 0.6*E
        theta = np.arctan2((p2-0.5), (p1-0.5))
        mode  = int(np.floor(((theta+np.pi)/(2*np.pi))*7)) % 7  # 0..6 对应 7 个调式
        return {"bpm": bpm, "swing": float(swing), "hat": float(hat), "kick": float(kick),
                "lead": float(lead), "bass": float(bass), "mode": int(mode)}

    persona = {}
    for n in classes:
        persona[n] = {"metrics_raw": raw[n], "metrics_norm": norm[n], "music": to_params(n)}

    out = {
        "classes": classes,
        "z_dim": int(z_dim),
        "formulas_version": "v1.0",
        "persona": persona
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"✓ wrote {OUT_JSON}")

if __name__ == "__main__":
    main()
