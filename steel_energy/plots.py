"""Generate shareable figures from saved, measured results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from steel_energy.config import output_root, read_json


def export_figures() -> None:
    """Create charts without refitting models or recalculating selection."""
    root = output_root()
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    results = pd.read_csv(root / "end_of_day/experiment-results.csv").sort_values("mae")
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    ax.bar(
        results.id,
        results.mae,
        color=["#af7930" if k == "baseline" else "#214c61" for k in results.kind],
    )
    ax.set(
        ylabel="Development MAE (kWh / interval)",
        xlabel="Predeclared experiment",
        title="Three chronological folds | July-September 2018",
    )
    ax.grid(axis="y", alpha=0.18)
    fig.savefig(destination / "development.png", dpi=200)
    plt.close(fig)
    predictions = pd.read_csv(
        root / "end_of_day/test-predictions.csv", parse_dates=["interval_start"]
    )
    week = predictions.loc[predictions.interval_start.lt("2018-11-08")]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    ax.plot(
        week.interval_start, week.energy, label="Observed", color="#243642", linewidth=1
    )
    ax.plot(
        week.interval_start,
        week.prediction,
        label="Frozen model",
        color="#b17628",
        linewidth=0.9,
        alpha=0.85,
    )
    ax.set(
        ylabel="Interval energy (kWh)",
        title="First locked-test week | no model refitting",
    )
    ax.legend(frameon=False, ncol=2)
    fig.savefig(destination / "test-week.png", dpi=200)
    plt.close(fig)
    literal = root / "literal/experiment-results.csv"
    if literal.exists():
        other = pd.read_csv(literal)
        compare = (
            results.set_index("id")[["mae"]]
            .join(
                other.set_index("id")[["mae"]], lsuffix="_primary", rsuffix="_literal"
            )
            .sort_index()
        )
        compare.to_csv(root / "timestamp-sensitivity.csv")
        fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
        compare.plot.bar(ax=ax, color=["#214c61", "#b17628"], rot=0)
        ax.set(
            ylabel="Development MAE (kWh)",
            xlabel="Configuration",
            title="Timestamp interpretation sensitivity | development only",
        )
        ax.legend(["End-of-day convention", "Literal timestamps sorted"], frameon=False)
        fig.savefig(destination / "timestamp-sensitivity.png", dpi=200)
        plt.close(fig)
    final = read_json(root / "end_of_day/final-results.json")
    fig, ax = plt.subplots(figsize=(7.5, 3), layout="constrained")
    a = final["advisory"]
    ax.barh(
        ["Correct high-load alerts", "False alarms", "Missed high-load intervals"],
        [a["true_positives"], a["false_positives"], a["false_negatives"]],
        color=["#214c61", "#b17628", "#a44d3a"],
    )
    ax.set(
        xlabel="15-minute intervals", title="Frozen advisory rule on the locked test"
    )
    fig.savefig(destination / "advisory.png", dpi=200)
    plt.close(fig)
