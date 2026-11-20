"""ICONN-2027 / Topic 1
Explainable machine learning for experimental band-gap prediction in inorganic semiconductors.

Data (real, public, unmodified):
  * Zhuo, Mansouri Tehrani & Brgoch, J. Phys. Chem. Lett. 9, 1668 (2018) experimental band gaps,
    distributed by matminer as `expt_gap` (6,354 rows).
  * JARVIS-DFT 3D (Choudhary et al., arXiv:2007.01831) OptB88vdW / TBmBJ gaps, used only as
    low-fidelity *features* for compounds that exist in both databases (multi-fidelity study).

Enhancements taken from the literature in ../papers (see README.md for the mapping):
  E1 two-stage metal/non-metal classifier + regressor            (Zhuo 2018; arXiv:2311.13778)
  E2 SHAP global/local explanations + SHAP-ranked feature subsets (arXiv:2503.04492, 2101.04383)
  E3 stacked ensemble of tree learners                            (arXiv:2407.18847)
  E4 leakage-free chemical-system GroupKFold evaluation           (arXiv:2604.25568)
  E5 multi-fidelity / Delta-learning with DFT gaps as inputs      (arXiv:2005.04338, 2611.10889)
  E6 physics-informed descriptors (ionic character, dX, ...)      (arXiv:2503.04492, 2501.02932)
No synthetic, oversampled or imputed *target* values are used anywhere.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(TOPIC))
from common.features import (DATA_RAW, chem_system, composition_features, parity_plot,  # noqa: E402
                             reduced_formula, regression_metrics, setup_plots)

RES = os.path.join(TOPIC, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
plt = setup_plots()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- models
def make_models():
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    imp = lambda: SimpleImputer(strategy="median")  # feature-level only
    rf = make_pipeline(imp(), RandomForestRegressor(n_estimators=400, min_samples_leaf=1,
                                                    max_features=0.33, n_jobs=-1, random_state=SEED))
    et = make_pipeline(imp(), ExtraTreesRegressor(n_estimators=400, max_features=0.5,
                                                  n_jobs=-1, random_state=SEED))
    xgb = XGBRegressor(n_estimators=900, learning_rate=0.03, max_depth=7, subsample=0.8,
                       colsample_bytree=0.6, min_child_weight=2, n_jobs=-1, random_state=SEED)
    lgbm = LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=63, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.6, min_child_samples=10,
                         random_state=SEED, verbose=-1)
    ridge = make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    stack = StackingRegressor(
        estimators=[("rf", rf), ("et", et),
                    ("xgb", make_pipeline(imp(), XGBRegressor(**xgb.get_params()))),
                    ("lgbm", make_pipeline(imp(), LGBMRegressor(**lgbm.get_params())))],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=5, n_jobs=1)
    return {"Ridge": ridge, "RandomForest": rf, "ExtraTrees": et, "XGBoost": xgb,
            "LightGBM": lgbm, "Stacked": stack}


class TwoStage:
    """E1: classify metal (gap==0) vs non-metal, then regress gap for non-metals."""

    def __init__(self):
        from lightgbm import LGBMClassifier, LGBMRegressor
        self.clf = LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                  colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
                                  random_state=SEED, verbose=-1)
        self.reg = LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=63,
                                 colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
                                 min_child_samples=10, random_state=SEED, verbose=-1)

    def fit(self, X, y):
        y = np.asarray(y)
        self.clf.fit(X, (y > 0).astype(int))
        self.reg.fit(X[y > 0], y[y > 0])
        return self

    def predict_proba_nonmetal(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X):
        p = self.predict_proba_nonmetal(X)
        return np.where(p >= 0.5, self.reg.predict(X), 0.0)


# ----------------------------------------------------------------------------- data
def load_data():
    from matminer.datasets import load_dataset
    df = load_dataset("expt_gap", data_home=DATA_RAW)
    n_raw = len(df)
    df["rf"] = df["formula"].apply(reduced_formula)
    df = df.dropna(subset=["rf"])
    grp = df.groupby("rf")["gap expt"]
    agg = pd.DataFrame({"gap": grp.median(), "n_reports": grp.size(),
                        "spread": grp.max() - grp.min()}).reset_index()
    # duplicates whose literature values disagree by >1 eV are ambiguous -> removed, not averaged
    n_conflict = int((agg["spread"] > 1.0).sum())
    agg = agg[agg["spread"] <= 1.0].reset_index(drop=True)
    agg["chemsys"] = agg["rf"].apply(chem_system)
    info = {"rows_raw": n_raw, "unique_compositions": int(len(grp)),
            "removed_conflicting_duplicates(>1eV)": n_conflict, "rows_used": int(len(agg)),
            "metals(gap=0)": int((agg.gap == 0).sum()), "non_metals": int((agg.gap > 0).sum()),
            "gap_range_eV": [float(agg.gap.min()), float(agg.gap.max())],
            "unique_chemical_systems": int(agg.chemsys.nunique())}
    return agg, info


# ----------------------------------------------------------------------------- main
def main():
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, train_test_split

    out = {}
    df, info = load_data()
    out["data"] = info
    log("data", info)
    X = composition_features(df["rf"], cache_name="t1_expt_gap_feats")
    keep = X.columns[X.notna().mean() > 0.9]
    X = X[keep]
    X = X.loc[:, X.std() > 0]
    y = df["gap"].values
    out["n_features"] = int(X.shape[1])
    log("features", X.shape)

    # ---------------- random hold-out (stratified on metal flag)
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED,
                                      stratify=(y > 0).astype(int))
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    models = make_models()
    holdout, preds = {}, {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, ytr)
        p = np.clip(m.predict(Xte), 0, None)
        preds[name] = p
        holdout[name] = {**regression_metrics(yte, p), "fit_seconds": round(time.time() - t0, 1)}
        log("holdout", name, holdout[name]["MAE"])
    ts = TwoStage().fit(Xtr, ytr)
    p = ts.predict(Xte)
    preds["TwoStage(E1)"] = p
    holdout["TwoStage(E1)"] = regression_metrics(yte, p)
    pn = ts.predict_proba_nonmetal(Xte)
    out["metal_classifier_holdout"] = {
        "accuracy": float(accuracy_score(yte > 0, pn >= 0.5)),
        "F1_nonmetal": float(f1_score(yte > 0, pn >= 0.5)),
        "ROC_AUC": float(roc_auc_score(yte > 0, pn))}
    # metrics restricted to semiconductors/insulators (the practically relevant subset)
    nm = yte > 0
    holdout_nm = {k: regression_metrics(yte[nm], v[nm]) for k, v in preds.items()}
    out["holdout_all"] = holdout
    out["holdout_nonmetals_only"] = holdout_nm
    pd.DataFrame(holdout).T.to_csv(os.path.join(RES, "table_holdout_all.csv"))
    pd.DataFrame(holdout_nm).T.to_csv(os.path.join(RES, "table_holdout_nonmetals.csv"))
    best = min(holdout, key=lambda k: holdout[k]["MAE"])
    out["best_holdout_model"] = best
    parity_plot(plt, yte, preds[best], os.path.join(FIG, "parity_best_holdout.png"),
                f"Experimental band gap – {best} (20% hold-out)", label="gap")
    pd.DataFrame({"formula": df["rf"].iloc[idx_te].values, "gap_expt": yte,
                  **{f"pred_{k}": v for k, v in preds.items()}}).to_csv(
        os.path.join(RES, "predictions_holdout.csv"), index=False)

    # ---------------- E4 leakage-free chemical-system GroupKFold (5 folds)
    from xgboost import XGBRegressor
    gkf = GroupKFold(n_splits=5)
    oof = {"XGBoost": np.zeros(len(y)), "TwoStage(E1)": np.zeros(len(y))}
    for tr, te in gkf.split(X, y, groups=df["chemsys"]):
        m = XGBRegressor(**models["XGBoost"].get_params()).fit(X.iloc[tr], y[tr])
        oof["XGBoost"][te] = np.clip(m.predict(X.iloc[te]), 0, None)
        oof["TwoStage(E1)"][te] = TwoStage().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    out["groupkfold_chemsys"] = {k: regression_metrics(y, v) for k, v in oof.items()}
    log("groupkfold", {k: round(v["MAE"], 3) for k, v in out["groupkfold_chemsys"].items()})
    parity_plot(plt, y, oof["XGBoost"], os.path.join(FIG, "parity_groupkfold_xgb.png"),
                "XGBoost, chemical-system GroupKFold (5-fold)", label="gap")

    # ---------------- E2 SHAP explanations (computed on TRAIN to rank, shown on TEST)
    import shap
    xgb = models["XGBoost"]
    expl = shap.TreeExplainer(xgb)
    sv_te = expl.shap_values(Xte)
    sv_tr = expl.shap_values(Xtr)
    imp = pd.Series(np.abs(sv_tr).mean(0), index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(RES, "shap_mean_abs_train.csv"), header=["mean_abs_shap"])
    plt.figure()
    shap.summary_plot(sv_te, Xte, max_display=20, show=False)
    plt.title("SHAP summary (XGBoost, hold-out set)", fontsize=10)
    plt.savefig(os.path.join(FIG, "shap_beeswarm.png")); plt.close("all")
    plt.figure()
    shap.summary_plot(sv_te, Xte, plot_type="bar", max_display=20, show=False)
    plt.savefig(os.path.join(FIG, "shap_bar.png")); plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
    for ax, f in zip(axes.ravel(), imp.index[:4]):
        ax.scatter(Xte[f], sv_te[:, X.columns.get_loc(f)], s=5, alpha=0.5, c=yte, cmap="viridis")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(f, fontsize=8); ax.set_ylabel("SHAP value (eV)")
    fig.suptitle("SHAP dependence of the four most influential descriptors", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "shap_dependence_top4.png")); plt.close(fig)

    # local explanation for three real hold-out compounds (wide-, mid-, narrow-gap)
    te_df = df.iloc[idx_te].reset_index(drop=True)
    picks = {}
    for lbl, target in [("wide", 5.0), ("mid", 2.0), ("narrow", 0.6)]:
        i = int(np.argmin(np.abs(yte - target)))
        picks[lbl] = i
        plt.figure()
        shap.plots.waterfall(shap.Explanation(sv_te[i], expl.expected_value, Xte.iloc[i].values,
                                              feature_names=list(X.columns)), max_display=10, show=False)
        plt.title(f"{te_df.rf[i]}: expt {yte[i]:.2f} eV, pred {preds['XGBoost'][i]:.2f} eV", fontsize=9)
        plt.savefig(os.path.join(FIG, f"shap_waterfall_{lbl}_{te_df.rf[i]}.png")); plt.close("all")
    out["local_explanations"] = {k: te_df.rf[v] for k, v in picks.items()}

    # ---------------- E2b SHAP-ranked compact models
    sel = {}
    for k in [5, 10, 15, 20, 30, 50, 80, X.shape[1]]:
        cols = imp.index[:k]
        m = XGBRegressor(**xgb.get_params()).fit(Xtr[cols], ytr)
        sel[int(k)] = regression_metrics(yte, np.clip(m.predict(Xte[cols]), 0, None))
    out["shap_feature_subsets"] = sel
    pd.DataFrame(sel).T.to_csv(os.path.join(RES, "table_shap_feature_subsets.csv"))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ks = list(sel); ax.plot(ks, [sel[k]["MAE"] for k in ks], "o-", color="#c05621")
    ax.set_xscale("log"); ax.set_xlabel("Number of SHAP-ranked descriptors"); ax.set_ylabel("Hold-out MAE (eV)")
    ax.set_title("Compact explainable models", fontsize=10)
    fig.savefig(os.path.join(FIG, "shap_topk_mae.png")); plt.close(fig)

    # ---------------- E5 multi-fidelity: JARVIS DFT gaps as features (overlap subset only)
    j = pd.read_pickle(os.path.join(DATA_RAW, "jarvis_dft_3d_noatoms.pkl"))
    j = j[["formula", "optb88vdw_bandgap", "mbj_bandgap", "formation_energy_peratom", "ehull"]].copy()
    for c in j.columns[1:]:
        j[c] = pd.to_numeric(j[c], errors="coerce")
    j["rf"] = j["formula"].apply(reduced_formula)
    j = j.sort_values("ehull").drop_duplicates("rf")  # most stable polymorph per composition
    mf = df.merge(j[["rf", "optb88vdw_bandgap", "mbj_bandgap"]], on="rf", how="inner")
    mf = mf.dropna(subset=["optb88vdw_bandgap"]).reset_index(drop=True)
    Xm = X.loc[df.reset_index().set_index("rf").loc[mf.rf, "index"].values].reset_index(drop=True)
    ym = mf.gap.values
    res_mf = {"n_overlap": int(len(mf)), "n_with_mbj": int(mf.mbj_bandgap.notna().sum()),
              "raw_DFT_vs_expt": {"OptB88vdW": regression_metrics(ym, mf.optb88vdw_bandgap)}}
    mm = mf.mbj_bandgap.notna().values
    res_mf["raw_DFT_vs_expt"]["TBmBJ(subset)"] = regression_metrics(ym[mm], mf.mbj_bandgap[mm])
    gk = GroupKFold(n_splits=5)
    variants = {"composition_only": Xm,
                "composition+OptB88": Xm.assign(dft_optb88=mf.optb88vdw_bandgap.values),
                "composition+OptB88+mBJ": Xm.assign(dft_optb88=mf.optb88vdw_bandgap.values,
                                                    dft_mbj=mf.mbj_bandgap.values)}
    oof_mf = {}
    for name, XX in variants.items():
        o = np.zeros(len(ym))
        for tr, te in gk.split(XX, ym, groups=mf.chemsys):
            m = XGBRegressor(**xgb.get_params()).fit(XX.iloc[tr], ym[tr])
            o[te] = np.clip(m.predict(XX.iloc[te]), 0, None)
        oof_mf[name] = o
        res_mf[name] = regression_metrics(ym, o)
    # Delta-learning: learn (expt - OptB88) correction
    o = np.zeros(len(ym)); XX = variants["composition+OptB88"]
    delta = ym - mf.optb88vdw_bandgap.values
    for tr, te in gk.split(XX, ym, groups=mf.chemsys):
        m = XGBRegressor(**xgb.get_params()).fit(XX.iloc[tr], delta[tr])
        o[te] = np.clip(mf.optb88vdw_bandgap.values[te] + m.predict(XX.iloc[te]), 0, None)
    res_mf["Delta_learning(OptB88+correction)"] = regression_metrics(ym, o)
    out["multi_fidelity"] = res_mf
    log("multi-fidelity", {k: (v["MAE"] if isinstance(v, dict) and "MAE" in v else "") for k, v in res_mf.items()})
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    names = ["OptB88vdW (raw DFT)", "Composition only", "+ OptB88 feature", "+ OptB88 + mBJ", "Δ-learning"]
    vals = [res_mf["raw_DFT_vs_expt"]["OptB88vdW"]["MAE"], res_mf["composition_only"]["MAE"],
            res_mf["composition+OptB88"]["MAE"], res_mf["composition+OptB88+mBJ"]["MAE"],
            res_mf["Delta_learning(OptB88+correction)"]["MAE"]]
    ax.barh(names, vals, color=["#a0aec0", "#2b6cb0", "#2c7a7b", "#2f855a", "#c05621"])
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("MAE vs experiment (eV), chemical-system GroupKFold")
    ax.set_title(f"Multi-fidelity study (n = {len(mf)} compounds in both databases)", fontsize=9)
    fig.savefig(os.path.join(FIG, "multifidelity_mae.png")); plt.close(fig)

    # ---------------- error analysis
    e = np.abs(preds[best] - yte)
    bins = pd.cut(yte, [-0.01, 0, 1, 2, 3, 5, 20], labels=["0 (metal)", "0-1", "1-2", "2-3", "3-5", ">5"])
    ea = pd.DataFrame({"bin": bins, "abs_err": e}).groupby("bin", observed=True)["abs_err"].agg(["count", "mean", "median"])
    ea.to_csv(os.path.join(RES, "table_error_by_gap_range.csv"))
    out["error_by_gap_range"] = ea.round(4).to_dict(orient="index")

    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    log("done; best hold-out model:", best, holdout[best])


if __name__ == "__main__":
    main()
