"""Statistical + financial/OR analysis tabs (fork extension).

Loads the analysis bundle exported by the main repo's local/viewer_export.py
(wells.parquet / rows.parquet / meta.json) and adds four tabs:

  S1  Residual anatomy   per-well RMSE distribution, ECDF, per-row QQ plot
  F1  Portfolio          leg-improvement covariance + efficient frontier
  F2  Tail risk          VaR/CVaR, Pareto contribution, per-well drawdown
  F4  Slot selection     2-slot field-bootstrap Monte Carlo over candidates

Discipline: these views generate hypotheses only; adoption decisions stay with
the measurement instruments (field-out holdout / production chain).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

FG = "#d8d8d8"
ACCENT = "#f59e0b"
GOOD = "#4ade80"
BAD = "#f472b6"
BLUE = "#38bdf8"


class AnalysisBundle:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.wells = pd.read_parquet(self.root / "wells.parquet")
        self.rows = pd.read_parquet(self.root / "rows.parquet")
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.candidates: list[str] = self.meta["candidates"]
        self.legs: list[str] = [l for l in self.meta["legs"] if l != "const"]
        for c in self.candidates:
            self.wells[f"rmse_{c}"] = np.sqrt(self.wells[f"mse_{c}"])
        for l in self.meta["legs"]:
            self.wells[f"rmse_leg_{l}"] = np.sqrt(self.wells[f"mse_leg_{l}"])

    def pooled(self, mse_col: str, wells: Optional[pd.DataFrame] = None) -> float:
        w = self.wells if wells is None else wells
        return float(np.sqrt((w[mse_col] * w.n_rows).sum() / w.n_rows.sum()))


def _plot(title: str) -> pg.PlotWidget:
    p = pg.PlotWidget(title=title)
    p.showGrid(x=True, y=True, alpha=0.2)
    return p


class StatsTab(QtWidgets.QWidget):
    """S1: per-well RMSE distribution + ECDF + per-row residual QQ."""

    def __init__(self, bundle: AnalysisBundle):
        super().__init__()
        self.b = bundle
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("candidate:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(bundle.candidates)
        self.combo.setCurrentText(bundle.candidates[-1])
        self.combo.currentTextChanged.connect(self.refresh)
        top.addWidget(self.combo)
        self.lbl = QtWidgets.QLabel("")
        top.addWidget(self.lbl)
        top.addStretch(1)
        lay.addLayout(top)
        grid = QtWidgets.QGridLayout()
        self.hist = _plot("per-well RMSE histogram (by field color)")
        self.ecdf = _plot("per-well RMSE ECDF (all candidates)")
        self.qq = _plot("per-row residual QQ vs Normal / Laplace")
        grid.addWidget(self.hist, 0, 0)
        grid.addWidget(self.ecdf, 0, 1)
        grid.addWidget(self.qq, 1, 0, 1, 2)
        lay.addLayout(grid)
        self.refresh()

    def refresh(self) -> None:
        c = self.combo.currentText()
        w = self.b.wells
        self.lbl.setText(f"  pooled={self.b.pooled(f'mse_{c}'):.3f}  "
                         f"median={w[f'rmse_{c}'].median():.2f}  "
                         f"wells={len(w)}")
        self.hist.clear()
        y, x = np.histogram(w[f"rmse_{c}"], bins=40)
        self.hist.addItem(pg.BarGraphItem(x0=x[:-1], x1=x[1:], height=y, brush=BLUE))
        self.ecdf.clear()
        colors = [ACCENT, GOOD, BAD, BLUE, "#a78bfa"]
        for k, cand in enumerate(self.b.candidates):
            v = np.sort(w[f"rmse_{cand}"].values)
            self.ecdf.plot(v, np.arange(1, len(v) + 1) / len(v),
                           pen=pg.mkPen(colors[k % len(colors)], width=2), name=cand)
        self.ecdf.addLegend()
        self.qq.clear()
        r = (self.b.rows[f"pred_{c}"] - self.b.rows.tvt_true).dropna()
        r = r.sample(min(len(r), 100_000), random_state=0).values
        r = np.sort((r - r.mean()) / r.std())
        n = len(r)
        q = (np.arange(1, n + 1) - 0.5) / n
        from scipy.stats import norm, laplace
        self.qq.plot(norm.ppf(q), r, pen=None, symbol="o", symbolSize=2,
                     symbolPen=None, symbolBrush=BLUE)
        lim = np.array([-5, 5])
        self.qq.plot(lim, lim, pen=pg.mkPen(GOOD, width=1.5))
        self.qq.plot(norm.ppf(q), laplace.ppf(q) / np.sqrt(2),
                     pen=pg.mkPen(BAD, width=1.5, style=QtCore.Qt.DashLine))
        self.qq.setLabel("bottom", "normal quantile")
        self.qq.setLabel("left", "standardized residual (green=Normal, pink=Laplace)")


class PortfolioTab(QtWidgets.QWidget):
    """F1: legs as assets — improvement covariance + efficient frontier."""

    def __init__(self, bundle: AnalysisBundle):
        super().__init__()
        self.b = bundle
        lay = QtWidgets.QHBoxLayout(self)
        w = bundle.wells
        legs = bundle.legs
        # asset "return" per well = rmse_const - rmse_leg (ft of improvement)
        base = w["rmse_leg_const"]
        self.R = pd.DataFrame({l: base - w[f"rmse_leg_{l}"] for l in legs})

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel(
            "improvement covariance (legs as assets; diversification value)"))
        cov = self.R.cov().values
        img = pg.ImageView()
        img.setImage(cov.T)
        img.ui.roiBtn.hide(); img.ui.menuBtn.hide()
        left.addWidget(img)
        tbl = QtWidgets.QTableWidget(len(legs), 3)
        tbl.setHorizontalHeaderLabels(["leg", "mean impr (ft)", "std"])
        for i, l in enumerate(legs):
            tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(l))
            tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{self.R[l].mean():.2f}"))
            tbl.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{self.R[l].std():.2f}"))
        tbl.resizeColumnsToContents()
        left.addWidget(tbl)
        lay.addLayout(left, 1)

        self.frontier = _plot("random-weight portfolios: mean vs std of per-well improvement")
        rng = np.random.default_rng(0)
        Wgt = rng.dirichlet(np.ones(len(legs)), 4000)
        port = self.R.values @ Wgt.T           # [wells, 4000]
        mu, sd = port.mean(0), port.std(0)
        self.frontier.plot(sd, mu, pen=None, symbol="o", symbolSize=3,
                           symbolPen=None, symbolBrush="#556")
        bw = self.b.meta.get("blend_weights", {})
        cur = np.array([bw.get("pf", .2), 0.0, bw.get("ww", .2), bw.get("unet", .15)])
        if cur.sum() > 0:
            cur = cur / cur.sum()
            p = self.R.values @ cur
            self.frontier.plot([p.std()], [p.mean()], pen=None, symbol="star",
                               symbolSize=18, symbolBrush=ACCENT)
        self.frontier.setLabel("bottom", "std of improvement (risk)")
        self.frontier.setLabel("left", "mean improvement (ft)  ★=current blend")
        lay.addWidget(self.frontier, 2)


class TailTab(QtWidgets.QWidget):
    """F2: VaR/CVaR + Pareto contribution + per-well drawdown."""

    def __init__(self, bundle: AnalysisBundle):
        super().__init__()
        self.b = bundle
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("candidate:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(bundle.candidates)
        self.combo.setCurrentText(bundle.candidates[-1])
        self.combo.currentTextChanged.connect(self.refresh)
        top.addWidget(self.combo)
        self.lbl = QtWidgets.QLabel("")
        top.addWidget(self.lbl); top.addStretch(1)
        lay.addLayout(top)
        mid = QtWidgets.QHBoxLayout()
        self.pareto = _plot("Pareto: cumulative share of pooled MSE by worst wells")
        mid.addWidget(self.pareto, 2)
        self.wl = QtWidgets.QListWidget()
        self.wl.currentTextChanged.connect(self._drawdown)
        mid.addWidget(self.wl, 1)
        lay.addLayout(mid)
        self.dd = _plot("drawdown: cumulative squared error along the well (all candidates)")
        lay.addWidget(self.dd)
        self.refresh()

    def refresh(self) -> None:
        c = self.combo.currentText()
        w = self.b.wells.copy()
        w["contrib"] = w[f"mse_{c}"] * w.n_rows
        w = w.sort_values("contrib", ascending=False)
        share = w.contrib.cumsum() / w.contrib.sum()
        self.pareto.clear()
        self.pareto.plot(np.arange(1, len(w) + 1), share.values,
                         pen=pg.mkPen(ACCENT, width=2))
        k20 = float(share.iloc[19]) if len(share) > 20 else float("nan")
        r = np.sort(w[f"rmse_{c}"].values)
        var90 = float(np.quantile(r, 0.90))
        cvar90 = float(r[r >= var90].mean())
        var95 = float(np.quantile(r, 0.95))
        cvar95 = float(r[r >= var95].mean())
        self.lbl.setText(f"  VaR90={var90:.1f}  CVaR90={cvar90:.1f}  "
                         f"VaR95={var95:.1f}  CVaR95={cvar95:.1f}  "
                         f"worst20 share={100 * k20:.0f}%")
        self.wl.blockSignals(True)
        self.wl.clear()
        for well in w.head(60)["well"]:
            self.wl.addItem(well)
        self.wl.blockSignals(False)

    def _drawdown(self, well: str) -> None:
        if not well:
            return
        g = self.b.rows[self.b.rows.well == well].sort_values("row_idx")
        self.dd.clear()
        colors = {"anchor": "#a78bfa", "v13": BLUE, "v14": GOOD, "v15": ACCENT}
        for c in self.b.candidates:
            se = (g[f"pred_{c}"] - g.tvt_true) ** 2
            self.dd.plot(np.arange(len(g)), se.cumsum().values,
                         pen=pg.mkPen(colors.get(c, FG), width=2), name=c)
        self.dd.addLegend()
        self.dd.setTitle(f"drawdown — well {well}")


class SlotTab(QtWidgets.QWidget):
    """F4: 2-slot selection via field-bootstrap Monte Carlo.

    Kaggle scores the better of the two final submissions on private; the
    field bootstrap resamples whole fields (cluster bootstrap) to mimic the
    unseen private field mix. Public LB is not an input.
    """

    def __init__(self, bundle: AnalysisBundle):
        super().__init__()
        self.b = bundle
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("Run Monte Carlo (B=4000)")
        self.btn.clicked.connect(self.run)
        top.addWidget(self.btn)
        self.note = QtWidgets.QLabel(
            "cluster bootstrap over fields; pair score = min(slot A, slot B) per draw")
        top.addWidget(self.note); top.addStretch(1)
        lay.addLayout(top)
        self.tbl = QtWidgets.QTableWidget()
        lay.addWidget(self.tbl)
        self.hist = _plot("pair-min pooled RMSE distribution (best pair highlighted)")
        lay.addWidget(self.hist)

    def run(self) -> None:
        b = self.b
        W = b.wells
        fields = W.field.unique()
        rng = np.random.default_rng(0)
        B = 4000
        cands = b.candidates
        # precompute per-field sums
        fs = {c: (W[f"mse_{c}"] * W.n_rows).groupby(W.field).sum() for c in cands}
        fn = W.groupby("field")["n_rows"].sum()
        draws = rng.choice(len(fields), size=(B, len(fields)), replace=True)
        pooled = {}
        for c in cands:
            num = np.asarray(fs[c].reindex(fields).values)[draws].sum(1)
            den = np.asarray(fn.reindex(fields).values)[draws].sum(1)
            pooled[c] = np.sqrt(num / den)
        pairs = [(a, bb) for i, a in enumerate(cands) for bb in cands[i:]]
        rows = []
        for a, bb in pairs:
            m = np.minimum(pooled[a], pooled[bb])
            rows.append((f"{a}+{bb}", m.mean(), np.quantile(m, 0.9),
                         m[m >= np.quantile(m, 0.9)].mean()))
        rows.sort(key=lambda r: r[1])
        self.tbl.setRowCount(len(rows))
        self.tbl.setColumnCount(4)
        self.tbl.setHorizontalHeaderLabels(
            ["pair", "mean min-RMSE", "P90", "CVaR90 (tail)"])
        for i, (name, mu, p90, cv) in enumerate(rows):
            for j, v in enumerate([name, f"{mu:.3f}", f"{p90:.3f}", f"{cv:.3f}"]):
                self.tbl.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self.tbl.resizeColumnsToContents()
        self.hist.clear()
        best = rows[0][0]
        a, bb = best.split("+")
        m = np.minimum(pooled[a], pooled[bb])
        y, x = np.histogram(m, bins=60)
        self.hist.addItem(pg.BarGraphItem(x0=x[:-1], x1=x[1:], height=y, brush=ACCENT))
        self.hist.setTitle(f"best pair: {best}  mean={m.mean():.3f}")


def make_tabs(bundle_dir: Path) -> list[tuple[str, QtWidgets.QWidget]]:
    b = AnalysisBundle(bundle_dir)
    return [
        ("S1 Residuals", StatsTab(b)),
        ("F1 Portfolio", PortfolioTab(b)),
        ("F2 Tail Risk", TailTab(b)),
        ("F4 Slot MC", SlotTab(b)),
    ]
