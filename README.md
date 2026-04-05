# ML Explorer 🤖

An interactive, browser-based guide to Machine Learning — no setup, no code required.
Tune hyperparameters, watch decision boundaries shift in real time, and build intuition for the algorithms that power modern AI.

**[▶ Open the live app](https://mlexplorer-cuzhkegkswbvxjiy4vfpno.streamlit.app/)**

---

## What is this?

ML Explorer is a self-contained learning tool that covers applied machine learning from the ground up — probability distributions, regression mathematics, classification algorithms, ensemble methods, and model evaluation. Every concept is paired with a live, interactive demo where you control the parameters and see results instantly, without writing a single line of code.

It is designed as a practical companion for anyone studying data science, econometrics, or machine learning who wants to go beyond textbook formulas and develop real intuition for how these algorithms behave.

---

## What will I learn?

### 📊 Supervised Learning
Five fully interactive demos:

| Topic | What you can explore |
|-------|----------------------|
| **Linear Regression** | Adjust slope, noise, and sample size. Toggle residual lines. Watch R² and RMSE update live. |
| **Logistic Regression** | Control class separation and regularisation strength. See the decision boundary shift in real time. |
| **Decision Tree** | Tune depth and splitting criterion (Gini / Entropy). Visualise how regions are carved out. |
| **Random Forest** | Set the number of trees, depth, and feature fraction. Compare against a single tree side-by-side. |
| **KNN & SVM** | Switch between algorithms. Tune K, kernel type, C, and gamma. |

### 🔍 Unsupervised Learning
Three demos covering structure discovery in unlabelled data:

| Topic | What you can explore |
|-------|----------------------|
| **K-Means Clustering** | Choose K, spread, and sample size. See Voronoi regions and use the elbow chart to pick optimal K. |
| **PCA** | Control the number of features and components. Inspect the scree plot and 2D projection. |
| **DBSCAN** | Switch between Moons, Circles, and Blobs datasets. Tune ε and min_samples to see noise points appear. |

### 📈 Probability & Regression
Three demos on the mathematical foundations of ML:

| Topic | What you can explore |
|-------|----------------------|
| **Probability Distributions** | Explore Normal, Binomial, Poisson, Exponential, and Uniform. Full parameter control with optional CDF overlay. |
| **Regression Types** | Compare OLS, Ridge, Lasso, and Polynomial on the same noisy dataset. Tune α, polynomial degree, and noise. |
| **Bias-Variance Tradeoff** | Increase polynomial degree and watch bias² vs variance decompose across bootstrap samples. |

### 🎮 Model Playground
A full end-to-end training environment — pick a dataset, pick a model, tune it, and evaluate it:

- **5 datasets** — Iris, Wine, Breast Cancer, Synthetic Classification, Synthetic Regression
- **9 models** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes, Ridge, Lasso
- **Hyperparameter controls** in the sidebar, tailored to whichever model you select
- **Evaluation outputs** — Accuracy, Precision, Recall, F1, Cross-validation score, Confusion Matrix, ROC / AUC curve, Feature Importance

---

## How do I use it?

Just open the app — no account, no installation, no code needed:

**[https://mlexplorer-cuzhkegkswbvxjiy4vfpno.streamlit.app/](https://mlexplorer-cuzhkegkswbvxjiy4vfpno.streamlit.app/)**

Use the sidebar to navigate between topics. Every page has controls on the left and a live chart on the right. Move a slider and the chart updates instantly.

---

## Running it locally

If you prefer to run the app on your own machine:

```bash
# 1. Clone the repository
git clone https://github.com/fedamohammadi/ml_explorer.git
cd ml_explorer

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Tech stack

| Library | Role |
|---------|------|
| [Streamlit](https://streamlit.io) | Web app framework and reactive UI |
| [Plotly](https://plotly.com/python/) | Interactive charts and visualisations |
| [scikit-learn](https://scikit-learn.org) | All ML models and datasets |
| [NumPy](https://numpy.org) | Numerical computing |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [SciPy](https://scipy.org) | Probability distribution functions |

---

## Project structure

```
ml_explorer/
│
├── app.py                          # Home page (entry point)
│
├── pages/
│   ├── 1_Supervised_Learning.py    # Linear Regression, Logistic, Tree, RF, KNN/SVM
│   ├── 2_Unsupervised_Learning.py  # K-Means, PCA, DBSCAN
│   ├── 3_Probability_Regression.py # Distributions, Regression types, Bias-Variance
│   └── 4_Model_Playground.py       # Full train/evaluate/compare UI
│
├── utils/
│   └── styles.py                   # Shared theme, colours, CSS helpers
│
├── .streamlit/
│   └── config.toml                 # Dark theme configuration
│
└── requirements.txt
```
