"""
K-NN Deney Paneli (Streamlit)

Çalıştırma:  streamlit run app.py

Sol panelden veri (hazır veri seti / CSV) ve K-NN parametreleri seçilir;
sağ tarafta eğitim/test doğruluğu, karar sınırı ve "k'ya göre doğruluk" grafiği
anında güncellenir.
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="K-NN Deney Paneli", layout="wide")

MIN_SAMPLES = 20      # bundan az satırla anlamlı deney yapılamaz
MAX_CLASSES = 30      # bundan fazla sınıf varsa hedef sütun büyük ihtimalle yanlış
GRID = 160            # 2D karar sınırı ızgara çözünürlüğü
GRID3 = 16            # 3D bölge bulutu: eksen başına nokta sayısı (16^3 = 4096)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
TEST_SIZE = 0.25      # eğitim/test bölmesi sabit
SPLIT_SEED = 42
METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]


# ----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ----------------------------------------------------------------------------
def fail(message: str):
    """Hata mesajını göster ve uygulamayı kontrollü şekilde durdur."""
    st.error(message)
    st.stop()


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """CSV'yi okur; ayracı (, ; tab) ve Türkçe kodlamayı otomatik dener."""
    for enc in ("utf-8-sig", "cp1254", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Dosya kodlaması okunamadı.")


def sync_k_from_slider():
    st.session_state["k_num"] = st.session_state["k_slider"]


def sync_k_from_number():
    st.session_state["k_slider"] = st.session_state["k_num"]


def knn_params(k, metric, p, weights):
    params = dict(n_neighbors=k, weights=weights, metric=metric)
    if metric == "minkowski":
        params["p"] = p
    return params


def build_model(k, metric, p, weights, scale):
    """Ölçekleme + K-NN. Scaler Pipeline içinde olduğu için sadece eğitim verisine fit edilir."""
    return Pipeline(
        [
            ("scaler", StandardScaler() if scale else "passthrough"),
            ("knn", KNeighborsClassifier(**knn_params(k, metric, p, weights))),
        ]
    )


@st.cache_data(show_spinner=False)
def accuracy_curve(X_tr, X_te, y_tr, y_te, ks, metric, p, weights, scale):
    """Her k için eğitim ve test doğruluğu."""
    if scale:
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    train_acc, test_acc = [], []
    for k in ks:
        knn = KNeighborsClassifier(**knn_params(k, metric, p, weights)).fit(X_tr, y_tr)
        train_acc.append(knn.score(X_tr, y_tr))
        test_acc.append(knn.score(X_te, y_te))
    return train_acc, test_acc


def prepare_projection(dim, X_tr, X_te, y_tr, y_te, features, model, k, metric, p, weights, scale):
    """Karar sınırı grafiği için dim (2 veya 3) boyutlu görünüm hazırlar.

    Özellik sayısı dim'e eşitse asıl model doğrudan çizilir; fazlaysa PCA ile dim boyuta
    indirilip ayrı bir K-NN eğitilir. Özellik sayısı dim'den azsa None döner.
    """
    n_feat = len(features)
    if n_feat < dim:
        return None
    if n_feat == dim:
        return dict(model=model, Xtr=X_tr, Xte=X_te, labels=[str(f) for f in features], note="")
    Xa, Xb = X_tr, X_te
    if scale:
        sc = StandardScaler().fit(X_tr)
        Xa, Xb = sc.transform(X_tr), sc.transform(X_te)
    pca = PCA(n_components=dim, random_state=0).fit(Xa)
    Ztr, Zte = pca.transform(Xa), pca.transform(Xb)
    plot_model = build_model(k, metric, p, weights, False).fit(Ztr, y_tr)
    note = (
        f" {n_feat} özellik PCA ile {dim} boyuta indirildi (varyansın %{pca.explained_variance_ratio_.sum() * 100:.1f}'i). "
        f"Grafik bu {dim} boyutla eğitilmiş ayrı bir modeli gösterir "
        f"(test doğruluğu %{plot_model.score(Zte, y_te) * 100:.1f}); yukarıdaki yüzdeler tüm özelliklerle hesaplanır."
    )
    return dict(model=plot_model, Xtr=Ztr, Xte=Zte, labels=[f"PC{i + 1}" for i in range(dim)], note=note)


# ----------------------------------------------------------------------------
# Grafik fonksiyonları
# ----------------------------------------------------------------------------
def plot_boundary(model, Xtr2, Xte2, ytr, yte, class_names, xlabel, ylabel):
    n_cls = len(class_names)
    colors = plt.cm.tab10(np.arange(n_cls) % 10)

    pts = np.vstack([Xtr2, Xte2])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.08 * np.where(hi - lo > 0, hi - lo, 1.0)
    xx, yy = np.meshgrid(
        np.linspace(lo[0] - pad[0], hi[0] + pad[0], GRID),
        np.linspace(lo[1] - pad[1], hi[1] + pad[1], GRID),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, levels=np.arange(n_cls + 1) - 0.5, colors=colors, alpha=0.25)
    for c in range(n_cls):
        m_tr, m_te = ytr == c, yte == c
        ax.scatter(Xtr2[m_tr, 0], Xtr2[m_tr, 1], color=colors[c], s=24,
                   edgecolor="white", linewidth=0.5, label=class_names[c])
        ax.scatter(Xte2[m_te, 0], Xte2[m_te, 1], color=colors[c], s=44, marker="^",
                   edgecolor="black", linewidth=0.8)
    ax.scatter([], [], marker="^", color="lightgray", edgecolor="black", label="Test noktaları")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best", ncol=2 if n_cls > 6 else 1)
    fig.tight_layout()
    return fig


def plot_boundary_3d(model, Xtr3, Xte3, ytr, yte, class_names, labels, show_cloud):
    """Etkileşimli 3D grafik: eğitim/test noktaları + (isteğe bağlı) yarı saydam bölge bulutu."""
    n_cls = len(class_names)
    color = lambda c: PALETTE[c % len(PALETTE)]
    fig = go.Figure()

    if show_cloud:
        pts = np.vstack([Xtr3, Xte3])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.08 * np.where(hi - lo > 0, hi - lo, 1.0)
        axes = [np.linspace(lo[i] - pad[i], hi[i] + pad[i], GRID3) for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        grid = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
        pred = model.predict(grid)
        for c in range(n_cls):
            m = pred == c
            if m.any():
                fig.add_trace(go.Scatter3d(
                    x=grid[m, 0], y=grid[m, 1], z=grid[m, 2], mode="markers",
                    marker=dict(size=3, color=color(c)), opacity=0.10,
                    hoverinfo="skip", showlegend=False,
                ))

    for c in range(n_cls):
        m_tr, m_te = ytr == c, yte == c
        fig.add_trace(go.Scatter3d(
            x=Xtr3[m_tr, 0], y=Xtr3[m_tr, 1], z=Xtr3[m_tr, 2], mode="markers",
            marker=dict(size=3.5, color=color(c), line=dict(width=0.5, color="white")),
            name=class_names[c], legendgroup=class_names[c],
        ))
        fig.add_trace(go.Scatter3d(
            x=Xte3[m_te, 0], y=Xte3[m_te, 1], z=Xte3[m_te, 2], mode="markers",
            marker=dict(size=5, symbol="diamond", color=color(c), line=dict(width=1, color="black")),
            name=f"{class_names[c]} (test)", legendgroup=class_names[c], showlegend=False,
        ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="markers",
        marker=dict(size=5, symbol="diamond", color="lightgray", line=dict(width=1, color="black")),
        name="Test noktaları",
    ))
    fig.update_layout(
        scene=dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2]),
        margin=dict(l=0, r=0, t=10, b=0), height=480,
        legend=dict(orientation="h", y=-0.02),
    )
    return fig


def plot_curve(ks, train_acc, test_acc, current_k):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ks, [a * 100 for a in train_acc], marker="o", ms=3, label="Eğitim doğruluğu")
    ax.plot(ks, [a * 100 for a in test_acc], marker="s", ms=3, label="Test doğruluğu")
    ax.axvline(current_k, color="gray", linestyle="--", linewidth=1, label=f"Seçili k = {current_k}")
    ax.set_xlabel("k (komşu sayısı)")
    ax.set_ylabel("Doğruluk (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Veri kaynağı bölümleri (sol panel)
# ----------------------------------------------------------------------------
def ready_dataset(name):
    loaders = {"Iris": load_iris, "Wine": load_wine, "Breast Cancer": load_breast_cancer}
    d = loaders[name](as_frame=True)
    df = d.frame.copy()
    df["sinif"] = df["target"].map(lambda i: str(d.target_names[i]))
    return df.drop(columns="target"), "sinif"


def choose_columns(df, default_target):
    cols = list(df.columns)
    idx = cols.index(default_target) if default_target in cols else len(cols) - 1
    target = st.sidebar.selectbox("Hedef (etiket) sütunu", cols, index=idx)
    numeric = [c for c in df.select_dtypes(include="number").columns if c != target]
    skipped = [c for c in cols if c != target and c not in numeric]
    if skipped:
        st.sidebar.warning(
            "Sayısal olmayan sütunlar özellik olarak kullanılamaz: "
            + ", ".join(map(str, skipped))
            + ". (Ondalık ayracı virgül ise noktaya çevirin.)"
        )
    features = st.sidebar.multiselect(
        "Özellik sütunları", numeric, default=numeric,
        help="İki özellik seçerseniz karar sınırı doğrudan çizilir; daha fazlasında PCA ile 2 boyuta indirilir.",
    )
    return target, features


# ----------------------------------------------------------------------------
# ANA AKIŞ
# ----------------------------------------------------------------------------
st.title("K-NN (K En Yakın Komşu) Deney Paneli")

# --- 1. Veri ----------------------------------------------------------------
st.sidebar.header("1 · Veri")
source = st.sidebar.radio("Veri kaynağı", ["Hazır veri seti", "CSV yükle"])

if source == "Hazır veri seti":
    name = st.sidebar.selectbox("Veri seti", ["Iris", "Wine", "Breast Cancer"])
    df, default_target = ready_dataset(name)
    target, features = choose_columns(df, default_target)
else:
    uploaded = st.sidebar.file_uploader("CSV dosyası seçin", type=["csv"])
    if uploaded is None:
        st.info("Başlamak için soldan bir CSV dosyası yükleyin ya da hazır bir veri seti seçin.")
        st.stop()
    try:
        df = read_csv_bytes(uploaded.getvalue())
    except Exception as exc:  # bozuk / boş dosya
        fail(f"CSV okunamadı: {exc}")
    if df.shape[1] < 2:
        fail("CSV en az 2 sütun içermeli (özellik + hedef). Ayraç karakterini kontrol edin.")
    target, features = choose_columns(df, df.columns[-1])

if not features:
    fail("En az bir sayısal özellik sütunu seçmelisiniz.")

# --- Temizleme / doğrulama --------------------------------------------------
data = df[features + [target]].replace([np.inf, -np.inf], np.nan)
n_before = len(data)
data = data.dropna()
if len(data) < n_before:
    st.warning(f"Eksik değer içeren {n_before - len(data)} satır çıkarıldı.")
if len(data) < MIN_SAMPLES:
    fail(f"Kullanılabilir satır sayısı çok az ({len(data)}). En az {MIN_SAMPLES} satır gerekli.")

raw_target = data[target]
if raw_target.nunique() > MAX_CLASSES:
    fail(
        f"'{target}' sütununda {raw_target.nunique()} farklı değer var; sınıflandırma için uygun "
        "görünmüyor (regresyon sütunu ya da kimlik numarası olabilir). Hedef sütunu kontrol edin."
    )
if raw_target.nunique() < 2:
    fail(f"'{target}' sütununda tek bir sınıf var. K-NN için en az 2 sınıf gerekli.")

if pd.api.types.is_numeric_dtype(raw_target):
    if (raw_target == raw_target.round()).all():
        raw_target = raw_target.round().astype(int)
else:
    raw_target = raw_target.astype(str)
encoder = LabelEncoder()
y = encoder.fit_transform(raw_target)
class_names = [str(c) for c in encoder.classes_]
X = data[features].to_numpy(dtype=float)

# --- Eğitim / test bölmesi (sabit: %75 / %25, random_state=42) ---------------
try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )
except ValueError:
    st.warning("Sınıflardan biri çok az örnek içerdiği için katmanlı (stratified) bölme yapılamadı; rastgele bölündü.")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)

# --- 2. K-NN parametreleri --------------------------------------------------
st.sidebar.header("2 · K-NN parametreleri")
k_max = len(X_tr)
for key in ("k_slider", "k_num"):
    st.session_state.setdefault(key, min(5, k_max))
    st.session_state[key] = int(min(max(st.session_state[key], 1), k_max))  # aralık dışına çıkmasın
st.sidebar.slider("k (komşu sayısı)", 1, k_max, key="k_slider", on_change=sync_k_from_slider,
                  help="Üst sınır eğitim verisi sayısıdır. Aşağıdaki kutuyla birbirine bağlıdır.")
st.sidebar.number_input("k değerini yaz (veya + / − kullan)", min_value=1, max_value=k_max, step=1,
                        key="k_num", on_change=sync_k_from_number)
k = int(st.session_state["k_slider"])
metric = st.sidebar.selectbox("Mesafe metriği", METRICS)
p = int(st.sidebar.number_input("p (yalnızca minkowski)", 1, 10, 2, disabled=metric != "minkowski",
                                help="p=1 Manhattan, p=2 Öklid, büyük p Chebyshev'e yaklaşır."))
weights = st.sidebar.selectbox("weights (oy ağırlığı)", ["uniform", "distance"])
scale = st.sidebar.checkbox("Ölçekleme uygula (StandardScaler)", value=True)

# --- Model eğitimi ----------------------------------------------------------
model = build_model(k, metric, p, weights, scale).fit(X_tr, y_tr)
train_acc = model.score(X_tr, y_tr)
test_acc = model.score(X_te, y_te)

n_feat = len(features)

# ----------------------------------------------------------------------------
# ÇIKTILAR
# ----------------------------------------------------------------------------
c1, c2 = st.columns(2)
c1.metric("Eğitim doğruluğu", f"%{train_acc * 100:.1f}")
c2.metric("Test doğruluğu", f"%{test_acc * 100:.1f}")

left, right = st.columns(2)

with left:
    st.subheader("Karar sınırı")
    can_3d = n_feat >= 3
    view = st.radio("Görünüm", ["2D", "3D"], horizontal=True, disabled=not can_3d,
                    help="3D görünüm için en az 3 özellik seçili olmalıdır.")
    dim = 3 if (view == "3D" and can_3d) else 2
    if not can_3d:
        st.caption("3D görünüm için en az 3 özellikli veri gerekir.")
    show_cloud = True
    if dim == 3:
        show_cloud = st.checkbox("Bölge bulutunu göster", value=True,
                                 help="Modelin uzayın her bölgesi için tahminini soluk noktalarla gösterir.")

    proj = prepare_projection(dim, X_tr, X_te, y_tr, y_te, features, model, k, metric, p, weights, scale)
    if proj is None:
        st.info("Karar sınırı grafiği için en az 2 özellik gerekir.")
    elif dim == 2:
        with st.spinner("Karar sınırı çiziliyor..."):
            fig = plot_boundary(proj["model"], proj["Xtr"], proj["Xte"], y_tr, y_te,
                                class_names, proj["labels"][0], proj["labels"][1])
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Daire: eğitim noktaları · Üçgen: test noktaları · Arka plan: modelin tahmin bölgeleri." + proj["note"])
    else:
        with st.spinner("3D grafik hazırlanıyor..."):
            fig3d = plot_boundary_3d(proj["model"], proj["Xtr"], proj["Xte"], y_tr, y_te,
                                     class_names, proj["labels"], show_cloud)
        st.plotly_chart(fig3d)
        st.caption("Daire: eğitim noktaları · Elmas: test noktaları"
                   + (" · Soluk bulutlar: modelin tahmin bölgeleri" if show_cloud else "")
                   + ". Fareyle döndürebilir, tekerlekle yakınlaştırabilirsiniz." + proj["note"])

with right:
    st.subheader("k'ya göre doğruluk")
    k_hi = min(len(X_tr), max(100, k))
    ks = tuple(int(v) for v in np.unique(np.linspace(1, k_hi, min(k_hi, 60)).round()))
    tr_curve, te_curve = accuracy_curve(X_tr, X_te, y_tr, y_te, ks, metric, p, weights, scale)
    fig = plot_curve(ks, tr_curve, te_curve, k)
    st.pyplot(fig)
    plt.close(fig)
    best = int(np.argmax(te_curve))
    st.caption(
        f"Bu aralıkta en yüksek test doğruluğu k={ks[best]} için %{te_curve[best] * 100:.1f} "
        "(eşitlikte en küçük k gösterilir)."
    )

st.subheader("Veri önizleme")
st.dataframe(data.head(50))
st.caption(f"İlk {min(50, len(data))} satır gösteriliyor (toplam {len(data)} satır). Yalnızca seçili özellik sütunları ve hedef sütun yer alır.")
