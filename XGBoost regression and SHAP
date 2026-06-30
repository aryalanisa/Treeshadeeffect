import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import string

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import xgboost as xgb
import shap 
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


df = pd.read_csv ("PHD/Newdataset.csv")
Y = df['Temperature_difference']

# Features: numeric + categorical (one-hot encode)
numeric_feats = ["Agriculture",
    "Forest",
    "Grassland",
    "Water",
    "CanopyCover",
    "Imperviousdensity",
    "MeanCanopyHeight",
    "CanopyH_cv_20",
    "CanopyH_p95p5_20",
    "Shannon_landcover",
    "Vegetation",
    "GreenImperviousRatio",
    "log_Distancetowater",
    "log_Distancetoforest",
     "Hour",
    ]
categorical_feats = ['Surface','Location','Year', 'Landuse']
X = pd.get_dummies(df[numeric_feats + categorical_feats], drop_first = False)
print(X.columns)

#XGBoost regression
seed = 42
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_rmse, outer_r2 = [], []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"\n=== Fold {fold+1} ===")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    # --- Inner optimization function ---
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "reg:squarederror",
            "n_jobs": 1
        }

        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(
            model,
            X_train, y_train,
            cv=3,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            n_jobs=1
        )
        return np.mean(scores)

    # --- Fix Optuna randomness ---
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=10, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        "random_state": seed,
        "seed": seed,
        "deterministic_histogram": True,
        "n_jobs": 1
    })

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Fold {fold+1} RMSE: {rmse:.4f}, R2: {r2:.4f}")
    outer_rmse.append(rmse)
    outer_r2.append(r2)

# --- Summary results ---
print("\nOuter CV RMSEs:", outer_rmse)
print("Mean RMSE:", np.mean(outer_rmse), "±", np.std(outer_rmse))
print("\nOuter CV R2s:", outer_r2)
print("Mean R2:", np.mean(outer_r2), "±", np.std(outer_r2))

# --- Plot ---
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Model Performance (XGBoost)")
plt.text(
    0.05, 0.95, f"$R^2$ = {r2:.2f}",
    transform=plt.gca().transAxes, fontsize=12,
    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
#plt.savefig("File/Modelperformance_new.tif", bbox_inches = 'tight', dpi = 300)
plt.show()

#SHAP analysis
feature_name_map = {
   "Agriculture": "Agriculture %",
    "Forest": "Forest %",
    "Grassland": "Grassland %",
    "Water": "Water %",
    "CanopyCover": "Tree cover density",
    "Imperviousdensity": "Imperviousness density",
    "MeanCanopyHeight": "Mean canopy height",
    "CanopyH_cv_20": "Variation in canopy height",
    "CanopyH_p95p5_20": "Height range of trees",
    "Shannon_landcover": "Landscape diversity",
    "Vegetation": "Total vegetation % in an area",
    "GreenImperviousRatio": "Green-Impervious ratio",
    "log_Distancetowater": "Distance to water",
    "log_Distancetoforest": "Distance to forest",
    "Hour": "Hour of the day"
      
}
X_plot = X.copy()
X_plot.columns = [feature_name_map.get(c, c) for c in X.columns]

# ==============================
# Train final model
# ==============================
best_params = study.best_params
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X, Y)

# ==============================
# SHAP computation
# ==============================
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_plot, check_additivity=False)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
rel_importance = 100 * mean_abs_shap / mean_abs_shap.sum()

rel_imp_df = (
    pd.Series(rel_importance, index=X_plot.columns)
    .sort_values(ascending=False)
)
top_n = 6
top_idx = np.argsort(mean_abs_shap)[::-1][:top_n] 

fig, axes = plt.subplots(
    1, 2, figsize=(20, 10),
    gridspec_kw={"width_ratios": [3, 3.5]}
)

# ---- (a) Relative importance bar plot ----
vals = rel_imp_df.head(top_n).sort_values()   # SAME relative importance values
ypos = np.arange(len(vals))

axes[0].barh(
    ypos,
    vals.values,
    color="dodgerblue",
    edgecolor="black",
    height=0.7
)

# Y-axis labels
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(vals.index)

# Add relative importance values on bars (RESTORED)
for i, v in enumerate(vals.values):
    axes[0].text(
        v + 0.5,                 # small offset to the right
        i,
        f"{v:.1f}%",             # <-- relative importance restored
        va="center",
        ha="left",
        fontsize=9,
        color="black"
    )

# Axis styling
axes[0].set_xlabel("Relative importance (%)", fontsize=12)
axes[0].grid(axis="x", alpha=0.3)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].tick_params(axis="both", labelsize=10)

# ---- REMOVE GRID ----
axes[0].grid(False) 

# ---- pick top 8 features by mean(|SHAP|) ----
  # indices of top features

X_top = X_plot.iloc[:, top_idx]
shap_top = shap_values[:, top_idx]

# ---- beeswarm (works with summary_plot across versions) ----
plt.sca(axes[1])  # only if you're using your subplot axis
shap.summary_plot(
    shap_top,
    X_top,
    max_display=top_n,
    show=False
)

# x label
axes[1].set_xlabel("SHAP value", fontsize=12)

# reduce feature-name font size
axes[1].tick_params(axis="y", labelsize=10)  # try 9 or 8 if you want smaller


# Find colorbar safely
for ax in fig.axes:
    if ax.get_ylabel() == "Feature value":
        cbar_ax = ax
        break

pos = cbar_ax.get_position()
cbar_ax.set_position([
    pos.x0 + 0.07,  # ← spacing
    pos.y0,
    pos.width,
    pos.height
])
# Panel labels
fig = plt.gcf()

fig.text(0.02, 0.96, "(a)", fontsize=14, va="top")
fig.text(0.52, 0.96, "(b)", fontsize=14, va="top")


# Leave space on the right for colorbar
plt.tight_layout

# Increase gap between panels
plt.subplots_adjust(wspace=2.5, top=0.92)
plt.show()


#Location specific 
def collapse_landscape(row):
    if row["Landscape_Urban"] == 1:
        return "Urban"
    elif row["Landscape_Post_mine"] == 1:
        return "Semi-natural (Site A)"
    elif row["Landscape_Lakeside"] == 1:
        return "Semi-natural (Site B)"

X["Landscape"] = X.apply(collapse_landscape, axis=1)
feature_names = X.columns[:shap_values.shape[1]]
shap_df = pd.DataFrame(shap_values, columns=feature_names)
# Add Landscape back
shap_df["Landscape"] = X["Landscape"].values
# Mean absolute SHAP per feature per landscape
shap_landscape = (
    shap_df
    .groupby("Landscape")
    .apply(lambda df: df.abs().mean())
)
top_n = 5
landscapes = shap_landscape.index.tolist()

location_name_map = {
    "Semi-natural (Site A)": "Semi-natural (Cottbuser Ostsee)",
    "Semi-natural (Site B)": "Semi-natural (Sedlitzer See)",
    "Urban": "Urban (Cottbus)"
}

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(6, 12),
    sharey=False
)

panel_labels = list(string.ascii_lowercase)

for i, (ax, landscape) in enumerate(zip(axes, landscapes)):

    top_features = (
        shap_landscape.loc[landscape]
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Relative importance (%)
    top_features = 100 * top_features / top_features.sum()
    top_features = top_features.sort_values()

    plot_index = [
        feature_name_map.get(f, f) for f in top_features.index
    ]

    bars = ax.barh(
        plot_index,
        top_features.values,
        color="dodgerblue",
        edgecolor="black"
    )
    ax.tick_params(axis='both', colors='black', labelsize=14)

    ax.tick_params(axis="y", labelsize=14, color = 'black')

    # percentage labels inside bars
    for i_bar, value in enumerate(top_features.values):
        ax.text(
            value - 0.8,
            i_bar,
            f"{value:.1f}%",
            va="center",
            ha="right",
            color="white",
            fontsize=9,
            fontweight="bold"
        )

    # panel label
    ax.text(
        -0.1, 1.05,
        f"({panel_labels[i]})",
        transform=ax.transAxes,
        fontsize=12, color = 'black'
    )

    ax.set_title(location_name_map.get(landscape, landscape), fontsize=14, color="black")
    ax.set_xlabel("Relative importance (%)", fontsize=11,  color="black")
    ax.grid(axis="x", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# global title
plt.suptitle(
    "Location-specific variation in key predictors of ΔT",
    fontsize=16, color = 'black'
)

plt.tight_layout(rect=[-0.5, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.45)
plt.show()

#Landuse specific
def collapse_landuse(row):
    if row["Landuse_Forest"] == 1:
        return "Forest dominated region"
    elif row["Landuse_Grass"] == 1:
        return "Grassland dominated region"
    elif row["Landuse_Water"] == 1:
        return "Water dominated region"
    elif row["Landuse_Urban"] == 1:
        return "Urban dominated region"

X["Landuse"] = X.apply(collapse_landuse, axis=1)
feature_names = X.columns[:shap_values.shape[1]]
shap_df = pd.DataFrame(shap_values, columns=feature_names)
# Add Landscape back
shap_df["Landuse"] = X["Landuse"].values
# Mean absolute SHAP per feature per landscape
shap_landuse = (
    shap_df
    .groupby("Landuse")
    .apply(lambda df: df.abs().mean())
)
top_n = 5
landuses = shap_landuse.index.tolist()

nrows = 2
ncols = int(np.ceil(len(landuses) / nrows))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(7*ncols, 4*nrows),
    sharex=False
)

axes = axes.flatten()
panel_labels = list(string.ascii_lowercase)

for i, (ax, landuse) in enumerate(zip(axes, landuses)):

    top_features = (
        shap_landuse.loc[landuse]
        .sort_values(ascending=False)
        .head(top_n)
    )

    # convert to relative %
    top_features = 100 * top_features / top_features.sum()
    top_features = top_features.sort_values()

    # map feature names
    plot_index = [
        feature_name_map.get(f, f) for f in top_features.index
    ]

    ax.barh(
        plot_index,
        top_features.values,
        color="dodgerblue",
        edgecolor="black"
    )
    ax.tick_params(axis='x', labelsize = 14, colors='black')
    ax.tick_params(axis="y", labelsize=14, color = 'black')

    # percentage labels
    for j, v in enumerate(top_features.values):
        ax.text(
            v - 0.8,
            j,
            f"{v:.1f}%",
            va="center",
            ha="right",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title(landuse, fontsize=12, color = 'black')
    ax.set_xlabel("Relative importance (%)", fontsize=12, color = 'black')
    ax.grid(axis="x", alpha=0.3)

    # panel labels (a), (b), ...
    ax.text(
        -0.1,
        1.05,
        f"({panel_labels[i]})",
        transform=ax.transAxes,
        fontsize=12, color = 'black'
    )

# remove unused axes
for ax in axes[len(landuses):]:
    ax.remove()

# global title
plt.suptitle(
    "Landcover-specific variation in key predictors of shade induced cooling",
    fontsize=16, color = 'black'
)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.subplots_adjust(wspace=0.70, hspace=0.45)
plt.show()

#SHAP dependence plot
# ===============================
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

# handle both numpy array and Explanation object
shap_array = shap_values if isinstance(shap_values, np.ndarray) else shap_values.values

# ===============================
# Features to plot
# ===============================
features = [
    "MeanCanopyHeight",
    "CanopyH_cv_20",
    "Shannon_landcover",
    "log_Distancetoforest",
    "log_Distancetowater",
    "GreenImperviousRatio"
]

feature_label_map = {
    "MeanCanopyHeight": "Mean canopy height",
    "CanopyH_cv_20": "Variation in canopy height",
    "Shannon_landcover": "Landscape diversity",
    "log_Distancetoforest": "Distance to forest",
    "log_Distancetowater": "Distance to water",
    "GreenImperviousRatio": "Green-impervious ratio"
}

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# choose percentiles for y-axis trimming
y_lower_pct = 1
y_upper_pct = 99

# ===============================
# Figure layout
# ===============================
fig = plt.figure(figsize=(11, 7.5))
gs = fig.add_gridspec(2, 3, wspace=0.4, hspace=0.4)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2])
]

# ===============================
# Plot each dependence panel
# ===============================
for ax, feature, label in zip(axes, features, panel_labels):

    x = X[feature].values
    y = shap_array[:, X.columns.get_loc(feature)]

    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)

    # Remove extreme value (=1) only for WaterInfluence
    if feature == "WaterInfluence":
        mask = mask & (x < 0.99)

    x_plot = x[mask]
    y_plot = y[mask]
    # per-feature color normalization
    vmin = np.nanpercentile(x_plot, 1)
    vmax = np.nanpercentile(x_plot, 99)

    if vmin == vmax:
        vmin = np.nanmin(x_plot)
        vmax = np.nanmax(x_plot)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # scatter
    ax.scatter(
        x_plot,
        y_plot,
        c=x_plot,
        cmap="RdBu_r",
        norm=norm,
        s=12,
        alpha=0.45,
        edgecolor="none",
        zorder=1
    )

    # LOWESS smoothing
    smoothed = lowess(y_plot, x_plot, frac=0.4, return_sorted=True)

    ax.plot(
        smoothed[:, 0],
        smoothed[:, 1],
        color="#08519c",
        linewidth=2.0,
        zorder=3
    )

    # residual-based band
    residuals = y_plot - np.interp(x_plot, smoothed[:, 0], smoothed[:, 1])
    band = np.std(residuals) * 0.6

    ax.fill_between(
        smoothed[:, 0],
        smoothed[:, 1] - band,
        smoothed[:, 1] + band,
        color="#08519c",
        alpha=0.12,
        zorder=2
    )

    # zero line
    ax.axhline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.6)

    # labels
    ax.set_xlabel(feature_label_map.get(feature, feature), fontsize=10)
    ax.set_ylabel("SHAP value", fontsize=10)

    # x-axis limits based on percentiles
    xmin = np.nanpercentile(x_plot, 1)
    xmax = np.nanpercentile(x_plot, 99)
    xpad = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # y-axis limits based on percentiles
    ymin = np.nanpercentile(y_plot, y_lower_pct)
    ymax = np.nanpercentile(y_plot, y_upper_pct)
    ypad = (ymax - ymin) * 0.08 if ymax > ymin else 0.1
    ax.set_ylim(ymin - ypad, ymax + ypad)

    # styling
    ax.tick_params(axis="both", labelsize=9, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.text(
        -0.14, 1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="left"
    )

# ===============================
# Shared colorbar
# ===============================
cmap = plt.cm.RdBu_r
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation="vertical",
    fraction=0.025,
    pad=0.04,
    aspect=30
)

cbar.set_ticks([])
cbar.ax.text(
    0.5, -0.03, "Low",
    transform=cbar.ax.transAxes,
    ha="center",
    va="top",
    fontsize=9
)
cbar.ax.text(
    0.5, 1.03, "High",
    transform=cbar.ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=9
)
cbar.set_label("Feature value", fontsize=10, labelpad=8)


plt.show()
