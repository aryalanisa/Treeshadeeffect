#Import libraries
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns


df = pd.read_csv("File/dataset.csv")

#Shapiro wilk test to  check normality of data

stat, p = shapiro(df["Temperature_difference"])
print("Shapiro-Wilk statistic:", stat)
print("p-value:", p)

#Levene test
groups = [
    group["Temperature_difference"].dropna()
    for name, group in df.groupby(["Location", "Landcover"])
]

stat, p = levene(*groups)

print("Levene's statistic:", stat)
print("p-value:", p)

#Two-way ANOVA

dv = "Temperature_difference"
A = "Location"
B = "Landcover"

# Clean data
df2 = df[[dv, A, B]].dropna().copy()

df2[A] = df2[A].astype("category")
df2[B] = df2[B].astype("category")

# Grand mean
y = df2[dv].to_numpy()
grand_mean = y.mean()

# Means
mean_A = df2.groupby(A)[dv].mean()
mean_B = df2.groupby(B)[dv].mean()
mean_AB = df2.groupby([A, B])[dv].mean()

# Counts
n_A = df2.groupby(A)[dv].size()
n_B = df2.groupby(B)[dv].size()
n_AB = df2.groupby([A, B])[dv].size()

# Total SS
SS_total = ((df2[dv] - grand_mean) ** 2).sum()

# SS for A
SS_A = sum(n_A[a] * (mean_A[a] - grand_mean) ** 2 for a in mean_A.index)

# SS for B
SS_B = sum(n_B[b] * (mean_B[b] - grand_mean) ** 2 for b in mean_B.index)

# SS for interaction
# SS_AB = sum over cells: n_ab * (mean_ab - mean_a - mean_b + grand_mean)^2
SS_AB = 0.0
for (a, b), m_ab in mean_AB.items():
    SS_AB += n_AB[(a, b)] * (m_ab - mean_A[a] - mean_B[b] + grand_mean) ** 2

# Error SS
SS_error = SS_total - SS_A - SS_B - SS_AB

# Degrees of freedom
a_levels = df2[A].cat.categories.size
b_levels = df2[B].cat.categories.size
N = len(df2)

df_A = a_levels - 1
df_B = b_levels - 1
df_AB = df_A * df_B
df_error = N - (a_levels * b_levels)
df_total = N - 1

# Mean squares
MS_A = SS_A / df_A if df_A > 0 else np.nan
MS_B = SS_B / df_B if df_B > 0 else np.nan
MS_AB = SS_AB / df_AB if df_AB > 0 else np.nan
MS_error = SS_error / df_error if df_error > 0 else np.nan

# F and p
F_A = MS_A / MS_error
F_B = MS_B / MS_error
F_AB = MS_AB / MS_error

p_A = 1 - f.cdf(F_A, df_A, df_error)
p_B = 1 - f.cdf(F_B, df_B, df_error)
p_AB = 1 - f.cdf(F_AB, df_AB, df_error)

anova = pd.DataFrame(
    {
        "SS": [SS_A, SS_B, SS_AB, SS_error, SS_total],
        "df": [df_A, df_B, df_AB, df_error, df_total],
        "MS": [MS_A, MS_B, MS_AB, MS_error, np.nan],
        "F":  [F_A,  F_B,  F_AB,  np.nan,  np.nan],
        "p":  [p_A,  p_B,  p_AB,  np.nan,  np.nan],
    },
    index=[A, B, f"{A}:{B}", "Residual", "Total"]
)

print("=== Two-way ANOVA (SciPy-based, classical SS) ===")
print(anova)

#Tukey HSD test

dv = "Temperature_difference"
A = "Location"
B = "Landcover"

# Prepare data
df_tukey = df[[dv, A, B]].dropna().copy()

# Tukey HSD
tukey = pairwise_tukeyhsd(
    endog=df_tukey[dv],      # response variable
    groups=df_tukey[A],      # factor
    alpha=0.05
)

print(tukey)

# ---- Create descriptive statistics table ----
desc_table = (
    df.groupby("Location")["Temperature_difference"]
    .agg(
        n="count",
        Mean="mean",
        SD="std"
    )
    .reset_index()
)

# ---- Round values for reporting ----
desc_table["Mean"] = desc_table["Mean"].round(2)
desc_table["SD"] = desc_table["SD"].round(2)

print(desc_table)

# ---- Create combined descriptive statistics table ----
desc_table = (
    df.groupby(["Location", "Landcover"])["Temperature_difference"]
    .agg(
        n="count",
        Mean="mean",
        SD="std"
    )
    .reset_index()
)

# ---- Round values ----
desc_table["Mean"] = desc_table["Mean"].round(2)
desc_table["SD"] = desc_table["SD"].round(2)

print(desc_table)


#Visualization 
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "savefig.dpi": 600,
})

# =====================================================
# VARIABLES
# =====================================================
dv = "Temperature_difference"
A = "Location"
B = "Landcover"

# =====================================================
# PREPARE DATA - TOP
# =====================================================
df_overall = df[[dv, A, B]].dropna().copy()

df_overall[A] = df_overall[A].replace({
    "Post_mine": "Semi-natural (Cottbuser Ostsee)",
    "Lakeside": "Semi-natural (Sedlitzer See)",
    "Urban": "Urban (Cottbus)"
})

overall_location_order = [
    "Urban (Cottbus)",
    "Semi-natural (Cottbuser Ostsee)",
    "Semi-natural (Sedlitzer See)"
]

df_overall = df_overall[df_overall[A].isin(overall_location_order)].copy()

# =====================================================
# PREPARE DATA - BOTTOM
# =====================================================
df_plot = df[[dv, A, B]].dropna().copy()

df_plot[A] = df_plot[A].replace({
    "Post_mine": "Semi-natural areas",
    "Lakeside": "Semi-natural areas",
    "Urban": "Urban"
})

location_order = ["Urban", "Semi-natural areas"]
landcover_order = ["Forest", "Grass", "Water", "Urban"]

df_plot = df_plot[df_plot[A].isin(location_order)].copy()
df_plot = df_plot[df_plot[B].isin(landcover_order)].copy()

# =====================================================
# COLORS
# =====================================================
palette = {
    "Forest": "#1b9e77",
    "Grass": "#66a61e",
    "Water": "#1f78b4",
    "Urban": "#d95f02"
}

location_palette = {
    "Urban (Cottbus)": "#d95f02",
    "Semi-natural (Cottbuser Ostsee)": "#7570b3",
    "Semi-natural (Sedlitzer See)": "#1f78b4"
}

# =====================================================
# SUMMARY STATS
# =====================================================
summary_top = (
    df_overall.groupby(A)[dv]
    .agg(mean="mean", sem="sem")
    .reindex(overall_location_order)
    .reset_index()
)

# =====================================================
# TUKEY TEST
# =====================================================
tukey = pairwise_tukeyhsd(
    endog=df_overall[dv],
    groups=df_overall[A],
    alpha=0.05
)

tukey_df = pd.DataFrame(
    tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

tukey_df["p-adj"] = tukey_df["p-adj"].astype(float)

def format_p(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"

# =====================================================
# FUNCTION FOR P-VALUE BRACKETS
# =====================================================
def add_sig_bracket(
    ax,
    x1,
    x2,
    y,
    h,
    text,
    text_offset=0.22,
    shrink=0.08
):


    x1s = x1 + shrink
    x2s = x2 - shrink

    ax.plot(
        [x1s, x1s, x2s, x2s],
        [y, y + h, y + h, y],
        color="black",
        lw=0.9,
        clip_on=False,
        zorder=20
    )

    ax.text(
        (x1 + x2) / 2,
        y + h + text_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
        zorder=25,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            pad=1.5
        )
    )

# =====================================================
# FIGURE SETUP
# =====================================================
sns.set_style("white")

fig = plt.figure(figsize=(9.5, 10))

gs = gridspec.GridSpec(
    2,
    2,
    height_ratios=[1.0, 1.35],
    hspace=0.40,
    wspace=0.28
)

ax_top = fig.add_subplot(gs[0, :])
ax_bl = fig.add_subplot(gs[1, 0])
ax_br = fig.add_subplot(gs[1, 1])

# =====================================================
# TOP PANEL
# =====================================================
sns.stripplot(
    data=df_overall,
    x=A,
    y=dv,
    order=overall_location_order,
    hue=A,
    palette=location_palette,
    jitter=0.15,
    alpha=0.25,
    size=2.2,
    ax=ax_top,
    zorder=1
)

if ax_top.get_legend():
    ax_top.get_legend().remove()

# Mean ± SEM
for i, row in summary_top.iterrows():
    ax_top.errorbar(
        i,
        row["mean"],
        yerr=row["sem"],
        fmt="o",
        color=location_palette[row[A]],
        markeredgecolor="black",
        markeredgewidth=1.1,
        markersize=9,
        capsize=5,
        elinewidth=1.8,
        capthick=1.8,
        zorder=10
    )

comparisons = [
    # shorter comparison placed lower
    ("Urban (Cottbus)", "Semi-natural (Cottbuser Ostsee)", 15.0),

    # shorter comparison placed in middle
    ("Semi-natural (Cottbuser Ostsee)", "Semi-natural (Sedlitzer See)", 16.7),

    # longest comparison placed highest
    ("Urban (Cottbus)", "Semi-natural (Sedlitzer See)", 18.4)
]

bracket_h = 0.18
text_offset = 0.22

for g1, g2, y in comparisons:
    row = tukey_df[
        ((tukey_df["group1"] == g1) & (tukey_df["group2"] == g2)) |
        ((tukey_df["group1"] == g2) & (tukey_df["group2"] == g1))
    ]

    if not row.empty:
        p = float(row["p-adj"].iloc[0])

        add_sig_bracket(
            ax_top,
            overall_location_order.index(g1),
            overall_location_order.index(g2),
            y,
            bracket_h,
            format_p(p),
            text_offset=text_offset,
            shrink=0.08
        )

ax_top.set_xlabel("Location", fontsize=13, labelpad=4, color="black")
ax_top.set_ylabel("ΔT ", fontsize=13, labelpad=6, color="black")

ax_top.tick_params(axis="both", labelsize=11, width=1.1, colors="black")

ax_top.text(
    -0.02, 1.05, "(a)",
    transform=ax_top.transAxes,
    fontsize=14,
    fontweight="bold",
    color="black"
)

# =====================================================
# BOTTOM PANELS
# =====================================================
for ax, location in zip([ax_bl, ax_br], location_order):

    sub = df_plot[df_plot[A] == location].copy()
    present_order = [lu for lu in landcover_order if lu in sub[B].unique()]

    for i, lu in enumerate(present_order):
        sub_lu = sub[sub[B] == lu]

        ax.boxplot(
            sub_lu[dv],
            positions=[i],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor="white",
                edgecolor=palette[lu],
                linewidth=1.7
            ),
            whiskerprops=dict(
                color=palette[lu],
                linewidth=1.7
            ),
            capprops=dict(
                color=palette[lu],
                linewidth=1.7
            ),
            medianprops=dict(
                color=palette[lu],
                linewidth=1.7
            )
        )

        x_jitter = np.random.normal(i, 0.06, len(sub_lu))

        ax.scatter(
            x_jitter,
            sub_lu[dv],
            color=palette[lu],
            alpha=0.38,
            s=28,
            zorder=3
        )

    ax.set_xticks(range(len(present_order)))
    ax.set_xticklabels(present_order, fontsize=11, color="black")

    ax.set_title(
        location,
        fontsize=13,
        fontweight="normal",
        color="black",
        pad=7
    )

    ax.set_xlabel("Landcover", fontsize=13, labelpad=5, color="black")
    ax.set_ylabel("ΔT ", fontsize=13, labelpad=6, color="black")

    ax.tick_params(axis="both", labelsize=11, width=1.1, colors="black")

    ax.grid(False)

# same y-limits for bottom panels
ax_bl.set_ylim(-10.5, 23.5)
ax_br.set_ylim(-10.5, 23.5)

# panel labels
ax_bl.text(
    -0.03, 1.02, "(b)",
    transform=ax_bl.transAxes,
    fontsize=14,
    fontweight="bold",
    color="black"
)

ax_br.text(
    -0.03, 1.02, "(c)",
    transform=ax_br.transAxes,
    fontsize=14,
    fontweight="bold",
    color="black"
)

# =====================================================
# SPINES
# =====================================================
for ax in [ax_top, ax_bl, ax_br]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

# =====================================================
# FINAL SPACING
# =====================================================
fig.subplots_adjust(
    left=0.09,
    right=0.985,
    bottom=0.08,
    top=0.97,
    hspace=0.42,
    wspace=0.30
)

# =====================================================
# SAVE
# =====================================================
#plt.savefig("PHD/ANOVAnew.tif", dpi=600, bbox_inches="tight")
#plt.savefig("PHD/ANOVAfinal.pdf", bbox_inches="tight")

plt.show()
