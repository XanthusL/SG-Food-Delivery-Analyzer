"""
Singapore Food Delivery Analyzer - Streamlit app
NVIDIA DLI mini project: Fundamentals of Accelerated Data Science

Zero-code-change GPU acceleration (from the course):
  - cudf.pandas  -> makes pandas run on the GPU
  - cuml.accel   -> makes scikit-learn run on the GPU
If there is no NVIDIA GPU (e.g. Streamlit Community Cloud), the same code runs on the CPU.
"""
# ---- Try to switch on GPU acceleration BEFORE importing pandas / sklearn ----
GPU_DATA, GPU_ML = False, False
try:
    import cudf.pandas
    cudf.pandas.install()
    GPU_DATA = True
except Exception:
    pass
try:
    import cuml.accel
    cuml.accel.install()
    GPU_ML = True
except Exception:
    pass

import time

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import xgboost as xgb
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="SG Food Delivery Analyzer", page_icon="🛵", layout="wide")

TOWNS = {  # name: (lat, long)
    "CBD": (1.283, 103.851), "Jurong": (1.333, 103.742), "Tampines": (1.354, 103.945),
    "Woodlands": (1.436, 103.786), "Bishan": (1.351, 103.848), "Punggol": (1.405, 103.902),
    "Clementi": (1.315, 103.765), "Bedok": (1.324, 103.930),
}
LAT0, LONG0 = 1.35, 103.82
FEATURES = ["hour", "is_raining", "items", "distance_km", "x_km", "y_km"]


# ---------- helpers ----------
def to_km(lat, long):
    """Lat/long are angles; convert to a flat x/y grid in km so distances make sense."""
    x = (long - LONG0) * 111.32 * np.cos(np.radians(lat))
    y = (lat - LAT0) * 110.57
    return x, y


def to_latlong(x, y):
    lat = y / 110.57 + LAT0
    long = x / (111.32 * np.cos(np.radians(lat))) + LONG0
    return lat, long


def timed(fn):
    start = time.time()
    out = fn()
    return out, time.time() - start


@st.cache_data(show_spinner="Generating orders...")
def make_orders(n, seed=42):
    """MOCK Singapore food delivery orders (randomly generated, not real data). Late deliveries are more likely in rain, at peak hours, and far away."""
    rng = np.random.default_rng(seed)
    names = np.array(list(TOWNS))
    centre = rng.integers(0, len(names), n)
    lat = np.array([TOWNS[t][0] for t in names])[centre] + rng.normal(0, 0.012, n)
    long = np.array([TOWNS[t][1] for t in names])[centre] + rng.normal(0, 0.012, n)

    weights = np.r_[[0.5] * 6, [2] * 5, [6] * 3, [3] * 4, [7] * 4, [1.5] * 2]
    hour = rng.choice(24, n, p=weights / weights.sum())
    peak = np.isin(hour, [11, 12, 13, 18, 19, 20, 21]).astype(int)
    rain = (rng.random(n) < 0.25).astype(int)
    items = rng.integers(1, 9, n)
    distance = np.abs(rng.normal(3.5, 1.8, n)).round(2)
    minutes = (12 + 3.2 * distance + 9 * rain + 6 * peak + 0.6 * items + rng.normal(0, 4, n)).round(1)
    late_p = 1 / (1 + np.exp(-(-5 + 0.45 * distance + 1.3 * rain + 1.0 * peak + 0.08 * items)))
    late = (rng.random(n) < late_p).astype(int)
    x, y = to_km(lat, long)

    return pd.DataFrame({
        "town": names[centre], "lat": lat, "lon": long, "x_km": x, "y_km": y,
        "hour": hour, "is_raining": rain, "items": items, "distance_km": distance,
        "delivery_min": minutes, "late": late,
    })


def shrink(df):
    """Memory management from the course: smallest dtype that fits + category for repeated text."""
    out = df.copy()
    for c in ["hour", "is_raining", "items", "late"]:
        out[c] = out[c].astype("int8")
    for c in ["lat", "lon", "x_km", "y_km", "distance_km", "delivery_min"]:
        out[c] = out[c].astype("float32")
    out["town"] = out["town"].astype("category")
    return out


@st.cache_resource(show_spinner="K-Means is placing kitchens...")
def fit_kitchens(n, k):
    df = make_orders(n)
    km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(df[["x_km", "y_km"]].sample(min(n, 200_000), random_state=0))
    centres = pd.DataFrame(km.cluster_centers_, columns=["x_km", "y_km"])
    centres["lat"], centres["lon"] = to_latlong(centres["x_km"].values, centres["y_km"].values)
    centres.index = [f"Kitchen {i + 1}" for i in range(k)]
    return centres


@st.cache_resource(show_spinner="Training XGBoost...")
def train_model(n):
    df = make_orders(n).sample(min(n, 300_000), random_state=1)
    X, y = df[FEATURES], df["late"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, tree_method="hist",
                              device="cuda" if GPU_ML else "cpu")
    _, secs = timed(lambda: model.fit(X_tr, y_tr))
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return model, auc, importance, secs, len(X_tr)


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    n_orders = st.select_slider("Number of orders", [50_000, 100_000, 250_000, 500_000, 1_000_000], value=250_000)
    st.divider()
    st.header("🖥️ Running on")
    st.write(f"**Data engine:** {'cuDF (GPU) via cudf.pandas' if GPU_DATA else 'pandas (CPU)'}")
    st.write(f"**ML engine:** {'cuML (GPU) via cuml.accel' if GPU_ML else 'scikit-learn (CPU)'}")
    st.info("This free server has no GPU, so everything runs on the CPU. On an NVIDIA GPU the same code "
            "is accelerated automatically. GPU speed tests are in the Colab notebook.")

orders = make_orders(n_orders)

st.title("🛵 Singapore Food Delivery Analyzer")
st.caption(f"Analysing **{len(orders):,} mock orders** across Singapore. "
           "NVIDIA DLI mini project: Fundamentals of Accelerated Data Science.")
st.warning("⚠️ **Mock data:** all orders are randomly generated for learning and demo purposes. "
           "They are not real customers, deliveries or company data.")

tabs = st.tabs(["📦 Data & memory", "📊 Late deliveries", "🍳 Kitchens (K-Means + KNN)",
                "🔥 Hotspots (DBSCAN)", "🤖 Predict late (XGBoost)", "🧠 How it works"])

# ----- Tab 1: data, memory, speed race -----
with tabs[0]:
    st.subheader("A peek at the data (mock data)")
    st.dataframe(orders.head(10), use_container_width=True)

    st.subheader("Memory management: shrink the data types")
    small = shrink(orders)
    before = orders.memory_usage(deep=True).sum() / 1e6
    after = small.memory_usage(deep=True).sum() / 1e6
    c1, c2, c3 = st.columns(3)
    c1.metric("Before (64-bit + text)", f"{before:.1f} MB")
    c2.metric("After (int8/float32 + category)", f"{after:.1f} MB")
    c3.metric("Saved", f"{(1 - after / before) * 100:.0f}%")
    st.caption("Ages, hours and 0/1 flags fit in int8 (1 byte instead of 8). Town names repeat, so `category` stores each name once.")

    st.subheader("Speed race: pandas vs Polars (lazy)")
    st.write("Same question for both: *late rate and average delivery time per town*.")
    if st.button("Run the race", type="primary"):
        _, t_pd = timed(lambda: orders.groupby("town").agg(late_rate=("late", "mean"), avg_min=("delivery_min", "mean")))
        pl_df = pl.from_pandas(orders[["town", "late", "delivery_min"]])
        query = pl_df.lazy().group_by("town").agg(pl.col("late").mean().alias("late_rate"),
                                                  pl.col("delivery_min").mean().alias("avg_min"))
        _, t_pl = timed(lambda: query.collect())
        race = pd.DataFrame({"seconds": [t_pd, t_pl]}, index=["pandas", "Polars (lazy)"])
        st.bar_chart(race)
        st.write(f"pandas: **{t_pd * 1000:.1f} ms** · Polars: **{t_pl * 1000:.1f} ms**")
        st.caption("Polars plans the whole query first (lazy execution), then runs it once in parallel. "
                   "On a GPU, `collect(engine='gpu')` and cuDF go even faster.")

# ----- Tab 2: analysis -----
with tabs[1]:
    st.write("**Rate trick:** the average of a 0/1 column = the percentage of 1s. So `late.mean()` is the late-delivery rate.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall late rate", f"{orders['late'].mean() * 100:.1f}%")
    c2.metric("Late rate when dry", f"{orders.loc[orders['is_raining'] == 0, 'late'].mean() * 100:.1f}%")
    c3.metric("Late rate when raining", f"{orders.loc[orders['is_raining'] == 1, 'late'].mean() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Late rate by town (groupby)")
        by_town = orders.groupby("town").agg(late_rate=("late", "mean"), avg_minutes=("delivery_min", "mean"),
                                            orders=("late", "size")).sort_values("late_rate", ascending=False)
        by_town["late_rate"] = (by_town["late_rate"] * 100).round(2)
        st.dataframe(by_town.round(1), use_container_width=True)
    with right:
        st.subheader("Late rate by hour: dry vs rain (pivot table)")
        pivot = orders.pivot_table(index="hour", columns="is_raining", values="late", aggfunc="mean") * 100
        pivot.columns = ["Dry %", "Rain %"]
        st.line_chart(pivot)

    st.subheader("Late rate by distance band (binning with cut)")
    bands = pd.cut(orders["distance_km"], bins=[0, 2, 4, 6, 50], labels=["0-2 km", "2-4 km", "4-6 km", "6+ km"])
    st.bar_chart(orders.groupby(bands, observed=True)["late"].mean() * 100)

    st.subheader("Where the orders are (5,000-point sample)")
    st.map(orders.sample(5000, random_state=0), latitude="lat", longitude="lon", size=20)

# ----- Tab 3: K-Means + KNN -----
with tabs[2]:
    k = st.slider("How many cloud kitchens?", 3, 10, 6)
    kitchens = fit_kitchens(n_orders, k)

    knn = NearestNeighbors(n_neighbors=1).fit(kitchens[["x_km", "y_km"]])
    (dist, idx), t_knn = timed(lambda: knn.kneighbors(orders[["x_km", "y_km"]]))
    nearest = pd.DataFrame({"kitchen": kitchens.index[idx.ravel()], "km": dist.ravel()})

    c1, c2 = st.columns(2)
    c1.metric("Average distance to nearest kitchen", f"{nearest['km'].mean():.2f} km")
    c2.metric("KNN lookups", f"{len(orders):,} in {t_knn:.2f} s")

    sample = orders.sample(4000, random_state=0)[["lat", "lon"]].assign(color="#9e9e9e", size=15)
    pins = kitchens[["lat", "lon"]].assign(color="#ff2b2b", size=400)
    st.map(pd.concat([sample, pins]), latitude="lat", longitude="lon", color="color", size="size")
    st.caption("Grey = customers, red = kitchen locations chosen by K-Means.")

    summary = nearest.groupby("kitchen").agg(orders=("km", "size"), avg_km=("km", "mean"))
    st.dataframe(summary.join(kitchens[["lat", "lon"]]).round(3), use_container_width=True)

# ----- Tab 4: DBSCAN -----
with tabs[3]:
    st.write("Find clusters of **late orders during rain**. DBSCAN doesn't need to know how many clusters there are. "
             "Points that don't belong to any cluster are noise (label -1).")
    late_rain = orders.loc[(orders["late"] == 1) & (orders["is_raining"] == 1)]
    late_rain = late_rain.sample(min(len(late_rain), 15_000), random_state=0).copy()
    default_ms = int(max(10, round(40 * len(late_rain) / 15_000, -1)))   # fewer points -> smaller hotspots

    c1, c2 = st.columns(2)
    eps = c1.slider("eps: max distance between neighbours (km)", 0.1, 1.0, 0.4, 0.05)
    min_samples = c2.slider("min_samples: smallest hotspot size", 10, 200, default_ms, 10)
    st.caption(f"Using {len(late_rain):,} late rainy orders.")
    labels, t_db = timed(lambda: DBSCAN(eps=eps, min_samples=min_samples).fit_predict(late_rain[["x_km", "y_km"]]))
    late_rain["cluster"] = labels

    found = late_rain[late_rain["cluster"] != -1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Hotspots found", int(found["cluster"].nunique()))
    c2.metric("Noise points", f"{(labels == -1).sum():,}")
    c3.metric("Time", f"{t_db:.2f} s")

    palette = ["#ff2b2b", "#ffa421", "#21c354", "#1c83e1", "#803df5", "#ff4b9b", "#00c0f2", "#ffe312"]
    shown = found.assign(color=found["cluster"].map(lambda c: palette[c % len(palette)]), size=25)
    if len(shown):
        st.map(shown, latitude="lat", longitude="lon", color="color", size="size")
        centres = found.groupby("cluster").agg(orders=("lat", "size"), lat=("lat", "mean"), lon=("lon", "mean"))
        st.dataframe(centres.sort_values("orders", ascending=False).round(4), use_container_width=True)
    else:
        st.warning("No hotspots with these settings. Try a bigger eps or a smaller min_samples.")

# ----- Tab 5: XGBoost -----
with tabs[4]:
    model, auc, importance, t_train, n_train = train_model(n_orders)
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC AUC on test data", f"{auc:.3f}")
    c2.metric("Training rows", f"{n_train:,}")
    c3.metric("Training time", f"{t_train:.1f} s")
    st.caption("AUC: 0.5 = random guessing, 1.0 = perfect at ranking late orders above on-time ones.")

    left, right = st.columns(2)
    with left:
        st.subheader("Which features matter most?")
        st.bar_chart(importance)
    with right:
        st.subheader("Try an order")
        hour = st.slider("Hour of day", 0, 23, 19)
        rain = st.toggle("Raining?", value=True)
        items = st.slider("Number of items", 1, 8, 3)
        distance = st.slider("Distance (km)", 0.5, 10.0, 4.0, 0.5)
        town = st.selectbox("Town", list(TOWNS))
        x, y = to_km(TOWNS[town][0], TOWNS[town][1])
        row = pd.DataFrame([[hour, int(rain), items, distance, x, y]], columns=FEATURES)
        p = float(model.predict_proba(row)[0, 1])
        st.metric("Chance this order is late", f"{p * 100:.1f}%")
        st.progress(p)

# ----- Tab 6: explanation -----
with tabs[5]:
    st.markdown("""
### Course concepts used in this app
| Tab | Concept from *Fundamentals of Accelerated Data Science* |
| --- | --- |
| Data & memory | DataFrames, `astype` to int8/float32, `category`, Polars lazy execution |
| Late deliveries | Boolean masks, `groupby`, mean-of-0/1 rate trick, `pivot_table`, binning with `cut` |
| Kitchens | Converting lat/long to km, **K-Means** clustering, **KNN** nearest neighbour |
| Hotspots | **DBSCAN** clustering with `eps` and `min_samples` |
| Predict late | **XGBoost**, train/test split, ROC AUC, feature importance |

### GPU acceleration with zero code changes
At the very top of `app.py`:
```python
import cudf.pandas;  cudf.pandas.install()   # pandas  -> GPU (cuDF)
import cuml.accel;   cuml.accel.install()    # sklearn -> GPU (cuML)
```
On a machine with an NVIDIA GPU and RAPIDS, every pandas and scikit-learn call below runs on the GPU automatically.
On this free server there is no GPU, so the same code runs on the CPU.

### Why a GPU helps
A CPU is a few very fast chefs; a GPU is thousands of kitchen helpers working at the same time.
Data work like "calculate the late rate for 1 million orders" is millions of small, identical tasks, perfect for the helpers.

### About the data
**This is mock data.** The orders are randomly generated by the app (not real customers or company data), clustered around 8 town centres. Late deliveries are made more likely
by rain, peak hours, distance and number of items, so the models have real patterns to find.
""")
