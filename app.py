# app.py — tanpa Debug tab, + tab "Informasi Fitur & Aplikasi"
import streamlit as st
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
try:
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
except Exception:
    OneHotEncoder = object  # type: ignore
    OrdinalEncoder = object  # type: ignore

# ---------- Page config + light styling ----------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.markdown("""
<style>
.big-title{font-size:2.2rem;font-weight:800}
.help{opacity:.75;font-size:.9rem}
.result-card{border-radius:16px;padding:18px 20px;border:1px solid rgba(250,250,250,.12)}
.cta button{border-radius:999px!important;padding:.65rem 1.2rem!important;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">❤️ Heart Disease Prediction App</div>', unsafe_allow_html=True)
st.caption("UI ini membaca struktur pipeline dari `final_pipe.sav` terbaru Anda — nama kolom, urutan, dan kategori valid diambil langsung dari model.")

# ---------- Load latest pipeline ----------
pipe = joblib.load("final_pipe.sav")

try:
    EXPECTED_COLS: List[str] = list(pipe.feature_names_in_)
except Exception:
    EXPECTED_COLS = []

# ---------- Inspect pipeline ----------
def find_column_transformer(p: Pipeline) -> ColumnTransformer | None:
    for _, step in getattr(p, "steps", []):
        if isinstance(step, ColumnTransformer):
            return step
    for step in getattr(p, "named_steps", {}).values():
        if isinstance(step, ColumnTransformer):
            return step
    for _, step in getattr(p, "steps", []):
        if isinstance(step, Pipeline):
            for _, sub in step.steps:
                if isinstance(sub, ColumnTransformer):
                    return sub
    return None

def collect_schema(ct: ColumnTransformer) -> Tuple[Dict[str, List[Any]], List[str]]:
    valid_cats: Dict[str, List[Any]] = {}
    numeric_cols: List[str] = []
    for _, trans, cols in ct.transformers_:
        if trans is None or trans == "drop":
            continue
        enc = trans
        if isinstance(trans, Pipeline):
            found = None
            for _, sub in trans.steps:
                if isinstance(sub, (OneHotEncoder, OrdinalEncoder)):
                    found = sub; break
            enc = found or trans
        if isinstance(enc, (OneHotEncoder, OrdinalEncoder)):
            cats = enc.categories_
            for i, c in enumerate(cols):
                valid_cats[c] = list(cats[i])
        else:
            numeric_cols.extend(list(cols))
    if EXPECTED_COLS:
        covered = set(valid_cats.keys()) | set(numeric_cols)
        for c in EXPECTED_COLS:
            if c not in covered:
                numeric_cols.append(c)
    # unique preserve order
    seen = set()
    numeric_cols = [c for c in numeric_cols if not (c in seen or seen.add(c))]
    return valid_cats, numeric_cols

ct = find_column_transformer(pipe)
if ct is None:
    st.error("❌ ColumnTransformer tidak ditemukan dalam pipeline. Pastikan model berisi preprocessing.")
    st.stop()

VALID_CATS, NUM_COLS = collect_schema(ct)
if not EXPECTED_COLS:
    EXPECTED_COLS = list(VALID_CATS.keys()) + [c for c in NUM_COLS if c not in VALID_CATS]

# --- heuristik batas + jenis kontrol (int/float) ---
def default_numeric_bounds(name: str):
    ln = name.lower()
    # (lo, hi, default, is_int)
    if "age" in ln:
        return 18, 100, 50, True
    if "resting_blood" in ln or "blood_pressure" in ln:
        return 70, 220, 120, True
    if "cholesterol" in ln:
        return 100, 600, 200, True
    if "max_heart_rate" in ln:
        return 60, 220, 150, True
    if "st_depression" in ln:
        return 0.0, 6.0, 0.0, False
    if "num_major_vessels" in ln or "major_vessels" in ln or "count" in ln:
        return 0, 3, 0, True
    # fallback float
    return 0.0, 1000.0, 0.0, False


def render_inputs() -> Dict[str, Any]:
    """Rapi: numerik 2 kolom, kategorikal 2 kolom. Integer di-slider tanpa koma."""
    row: Dict[str, Any] = {}

    # ===== Numerik (2 kolom) =====
    st.markdown("### ⚙️ Fitur Numerik")
    num_cols = [c for c in EXPECTED_COLS if c in NUM_COLS]
    n_left, n_right = st.columns(2, gap="large")

    # bagi rata ke dua kolom supaya tidak kepanjangan
    split_idx = (len(num_cols) + 1) // 2
    nums_left = num_cols[:split_idx]
    nums_right = num_cols[split_idx:]

    with n_left:
        for col in nums_left:
            lo, hi, default, is_int = default_numeric_bounds(col)
            if is_int:
                # slider integer → tanpa koma
                val = st.slider(col, min_value=int(lo), max_value=int(hi),
                                value=int(default), step=1)
            else:
                # gunakan number_input untuk float agar ringkas
                val = st.number_input(col, value=float(default))
            row[col] = val

    with n_right:
        for col in nums_right:
            lo, hi, default, is_int = default_numeric_bounds(col)
            if is_int:
                val = st.slider(col, min_value=int(lo), max_value=int(hi),
                                value=int(default), step=1)
            else:
                val = st.number_input(col, value=float(default))
            row[col] = val

    st.markdown('<div class="help">Nilai default diset wajar agar simulasi cepat.</div>', unsafe_allow_html=True)

    # ===== Kategorikal (2 kolom) =====
    st.markdown("### 🧩 Fitur Kategorikal")
    cat_cols = [c for c in EXPECTED_COLS if c in VALID_CATS]
    c_left, c_right = st.columns(2, gap="large")
    split_idx = (len(cat_cols) + 1) // 2
    cats_left = cat_cols[:split_idx]
    cats_right = cat_cols[split_idx:]

    with c_left:
        for col in cats_left:
            opts = VALID_CATS[col]
            idx = min(len(opts)//2, len(opts)-1)
            row[col] = st.selectbox(col, options=opts, index=idx)

    with c_right:
        for col in cats_right:
            opts = VALID_CATS[col]
            idx = min(len(opts)//2, len(opts)-1)
            row[col] = st.selectbox(col, options=opts, index=idx)

    return row


# ---------- Tabs (Debug dihapus) ----------
tab_pred, tab_info = st.tabs(["🧪 Input & Prediksi", "ℹ️ Informasi Fitur & Aplikasi"])

with tab_pred:
    row_values = render_inputs()
    X = pd.DataFrame([row_values]).reindex(columns=EXPECTED_COLS)

    st.markdown("#### ")
    st.markdown('<div class="cta">', unsafe_allow_html=True)
    clicked = st.button("🔮 Prediksi Sekarang!")
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked:
        try:
            proba = None
            try: proba = float(pipe.predict_proba(X)[0][1])
            except Exception: pass
            pred = pipe.predict(X)[0]

            c1, c2 = st.columns([0.6, 0.4], gap="large")
            with c1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Hasil Prediksi")
                label = str(pred)
                if isinstance(pred, (int, float)) and pred in [0, 1]:
                    label = "Berisiko" if int(pred) == 1 else "Tidak Berisiko"
                st.metric("Klasifikasi", label)
                if proba is not None:
                    st.progress(min(max(proba, 0.0), 1.0), text=f"Probabilitas positif: {proba:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.caption("📌 Rangkuman Input (urutan sesuai model)")
                st.dataframe(X.T.rename(columns={0: "value"}))
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

# ---------- Informasi Fitur & Aplikasi ----------
with tab_info:
    st.markdown("### 🧾 Informasi Fitur")
    # Deskripsi fitur (akan dipakai jika cocok; sisanya ditandai '-')
    FEATURE_DESC = {
        "age": "Usia pasien (tahun).",
        "sex": "Jenis kelamin pasien.",
        "dataset": "Sumber dataset klinis (mis. Cleveland/Hungary/Switzerland/VA).",
        "chest_pain_type": "Tipe nyeri dada (typical, atypical, non-anginal, asymptomatic).",
        "resting_blood_pressure": "Tekanan darah istirahat (mm Hg).",
        "cholesterol": "Serum kolesterol (mg/dl).",
        "fasting_blood_sugar": "Gula darah puasa > 120 mg/dl (ya/tidak).",
        "resting_electrocardiogram": "Hasil EKG istirahat (normal, ST-T abnormality, LVH).",
        "max_heart_rate_achieved": "Detak jantung maksimum tercapai (bpm).",
        "exercise_induced_angina": "Angina yang dipicu olahraga (ya/tidak).",
        "st_depression": "Depresi segmen ST relatif garis isoelektrik.",
        "st_slope": "Kemiringan segmen ST saat puncak exercise.",
        "num_major_vessels": "Jumlah pembuluh utama yang terlihat pada fluoroskopi.",
        "thalassemia": "Jenis thalassemia (normal, fixed defect, reversable defect).",
    }

    # siapkan tabel info fitur
    rows = []
    for col in EXPECTED_COLS:
        ftype = "Kategorikal" if col in VALID_CATS else "Numerik"
        desc = FEATURE_DESC.get(col, "-")
        # jika kategorikal, tampilkan daftar kategori singkat
        if col in VALID_CATS:
            cats = ", ".join(map(str, VALID_CATS[col][:6]))
            if len(VALID_CATS[col]) > 6:
                cats += ", ..."
        else:
            cats = "-"
        rows.append({"Fitur": col, "Tipe": ftype, "Kategori/Contoh Nilai": cats, "Deskripsi": desc})

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### ℹ️ Informasi Aplikasi")
    st.write(
        "- Aplikasi ini menggunakan **scikit-learn Pipeline + ColumnTransformer** yang tersimpan dalam `final_pipe.sav`.\n"
        "- UI otomatis mengikuti **urutan kolom** dan **kategori valid** dari model, sehingga aman dari mismatch.\n"
        "- Untuk menerima kategori baru tanpa error, latih ulang model dengan `OneHotEncoder(handle_unknown='ignore')` lalu simpan ulang pipeline.\n"
        "- Disarankan menyimpan metadata versi saat training untuk reprodusibilitas (contoh: `{'sklearn_version': sklearn.__version__}` bersama model)."
    )

st.divider()
st.caption("© 2025 • Heart Disease Prediction • Streamlit")




# cd /Users/malifhdyh/Downloads/Porto\ 3
# streamlit run app.py