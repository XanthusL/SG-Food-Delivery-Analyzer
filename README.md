# 🛵 Singapore Food Delivery Analyzer

An interactive data science app that analyses up to **1 million mock food delivery orders** across Singapore to answer real business questions:

- 📊 Where and when are deliveries late?
- 🍳 Where should we open cloud kitchens?
- 🔥 Where are the hotspots of late orders when it rains?
- 🤖 Can we predict if an order will be late?

> ⚠️ **Mock data:** all orders in this app are randomly generated for learning and demo purposes. They are not real customers, deliveries or company data.

Built as a mini project to apply what I learned in the **NVIDIA Deep Learning Institute** course **Fundamentals of Accelerated Data Science**.

🔗 **Live demo:**https://sg-food-delivery-analyzer.streamlit.app/**
---

## ✨ Features

| Tab | What it does | Technique |
| --- | --- | --- |
| 📦 **Data & memory** | Shrinks memory by choosing smaller data types, then races pandas against Polars | `astype`, `category`, Polars lazy execution |
| 📊 **Late deliveries** | Late rate by town, by hour (dry vs rain), and by distance, plus a map of orders | `groupby`, `pivot_table`, `cut`, boolean masks |
| 🍳 **Kitchens** | Choose 3–10 cloud kitchens; K-Means places them and KNN finds every order's nearest kitchen | **K-Means**, **K-Nearest Neighbors** |
| 🔥 **Hotspots** | Finds clusters of late rainy orders; tune `eps` and `min_samples` live | **DBSCAN** |
| 🤖 **Predict late** | Trains a model, shows AUC and feature importance, and predicts any order you set up | **XGBoost**, ROC AUC |
| 🧠 **How it works** | Plain-English explanation of every concept | — |

---

## ⚡ GPU acceleration with zero code changes

The first lines of `app.py` try to switch on NVIDIA RAPIDS:

```python
import cudf.pandas;  cudf.pandas.install()   # pandas  -> runs on GPU (cuDF)
import cuml.accel;   cuml.accel.install()    # sklearn -> runs on GPU (cuML)
```

- **On a machine with an NVIDIA GPU + RAPIDS:** every pandas and scikit-learn call runs on the GPU automatically.
- **On Streamlit Community Cloud (no GPU):** the same code runs on the CPU with normal pandas and scikit-learn.

The sidebar shows which engine is running.

> **Why GPUs help:** a CPU is a few very fast chefs; a GPU is thousands of kitchen helpers working at once. Data work like "find the late rate for 1 million orders" is millions of small, identical tasks, perfect for the helpers.

---

## 🔗 How it connects to the NVIDIA course

| Course topic | Where it appears in the app |
| --- | --- |
| cuDF and `cudf.pandas` | Zero-code-change GPU switch at the top of `app.py` |
| Memory management (`astype`, `category`) | Data & memory tab |
| Grouping, pivot tables, binning | Late deliveries tab |
| Converting lat/long to a km grid (like OSGB36 in the course) | `to_km()` function, used by every map and model |
| Polars and lazy execution | Speed race in the Data & memory tab |
| cuML: K-Means, DBSCAN, KNN | Kitchens and Hotspots tabs |
| XGBoost on GPU (`device="cuda"`) | Predict late tab |


---

## 🚀 Run it yourself

```bash
git clone https://github.com/XanthusL/SG-Food-Delivery-Analyzer.git
cd SG-Food-Delivery-Analyzer
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`.

---

## 📁 Files

```
SG-Food-Delivery-Analyzer/
├── app.py
├── requirements.txt
├── README.md
└── SG_Food_Delivery_Analyzer.ipynb
```

---

## 📝 About the data (mock data)

The orders are **mock data**, randomly generated inside the app so it runs anywhere with no downloads. Customers cluster around 8 town centres (CBD, Jurong, Tampines, Woodlands, Bishan, Punggol, Clementi, Bedok). Late deliveries are made more likely by rain, peak hours, distance and number of items, so the models have real patterns to discover.

---

## 🛠️ Built with

Python · Streamlit · pandas · Polars · scikit-learn · XGBoost · NVIDIA RAPIDS (cuDF, cuML)

---

## 🙏 Acknowledgements

- **NVIDIA Deep Learning Institute** for the course content
- **Republic Polytechnic**, C295 Special Project elective
