from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ListData:
    run_start_utc: datetime
    channel_hg: dict[int, np.ndarray]
    channel_lg: dict[int, np.ndarray]
    event_t_us: np.ndarray
    event_trg_id: np.ndarray
    event_nhits: np.ndarray


RUN_START_RE = re.compile(r"Run start time:\s*(.+?)\s*UTC")
RUN_NUM_RE = re.compile(r"Run n\.\s*(\d+)")
START_TIME_RE = re.compile(r"Start Time:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})")
STOP_TIME_RE = re.compile(r"Stop Time:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})")
ELAPSED_RE = re.compile(r"Elapsed time\s*=\s*([\d.]+)\s*s")
KV_RE = re.compile(r"^([A-Za-z0-9_\[\]]+)\s+(.+?)\s*(?:#.*)?$")


def parse_run_info(info_path: Path) -> dict[str, Any]:
    text = info_path.read_text(encoding="utf-8", errors="ignore")
    metadata: dict[str, Any] = {"config": {}}

    run_match = RUN_NUM_RE.search(text)
    if run_match:
        metadata["run_number"] = int(run_match.group(1))

    start_match = START_TIME_RE.search(text)
    if start_match:
        metadata["start_time_local"] = start_match.group(1)

    stop_match = STOP_TIME_RE.search(text)
    if stop_match:
        metadata["stop_time_local"] = stop_match.group(1)

    elapsed_match = ELAPSED_RE.search(text)
    if elapsed_match:
        metadata["elapsed_seconds_reported"] = float(elapsed_match.group(1))

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        if ":" in line:
            continue
        match = KV_RE.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        metadata["config"][key] = raw_value.strip()

    return metadata


def parse_list_file(list_path: Path) -> ListData:
    channel_hg: dict[int, list[int]] = {ch: [] for ch in range(64)}
    channel_lg: dict[int, list[int]] = {ch: [] for ch in range(64)}
    event_t_us: list[float] = []
    event_trg_id: list[int] = []
    event_nhits: list[int] = []

    run_start_utc: datetime | None = None

    with list_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("//"):
                start_match = RUN_START_RE.search(stripped)
                if start_match:
                    run_start_utc = datetime.strptime(
                        start_match.group(1), "%a %b %d %H:%M:%S %Y"
                    ).replace(tzinfo=timezone.utc)
                continue

            if stripped.startswith("Brd"):
                continue

            parts = stripped.split()
            if len(parts) == 7:
                _, ch_s, lg_s, hg_s, t_us_s, trg_s, nhits_s = parts
                ch = int(ch_s)
                lg = int(lg_s)
                hg = int(hg_s)
                event_t_us.append(float(t_us_s))
                event_trg_id.append(int(trg_s))
                event_nhits.append(int(nhits_s))
            elif len(parts) == 4:
                _, ch_s, lg_s, hg_s = parts
                ch = int(ch_s)
                lg = int(lg_s)
                hg = int(hg_s)
            else:
                continue

            channel_lg[ch].append(lg)
            channel_hg[ch].append(hg)

    if run_start_utc is None:
        raise ValueError(f"Could not find run start time in {list_path}")

    channel_hg_np = {ch: np.asarray(vals, dtype=np.int16) for ch, vals in channel_hg.items()}
    channel_lg_np = {ch: np.asarray(vals, dtype=np.int16) for ch, vals in channel_lg.items()}

    return ListData(
        run_start_utc=run_start_utc,
        channel_hg=channel_hg_np,
        channel_lg=channel_lg_np,
        event_t_us=np.asarray(event_t_us, dtype=np.float64),
        event_trg_id=np.asarray(event_trg_id, dtype=np.int32),
        event_nhits=np.asarray(event_nhits, dtype=np.int16),
    )


def parse_service_info(service_path: Path) -> pd.DataFrame:
    service_df = pd.read_csv(service_path, sep=r"\s+", engine="python")
    service_df["timestamp_utc"] = pd.to_datetime(service_df["TStampPC"], unit="ms", utc=True)
    return service_df


def estimate_mip_peak(adc_values: np.ndarray) -> tuple[float, float, float]:
    values = adc_values.astype(np.float64)
    if values.size < 100:
        return math.nan, math.nan, math.nan

    pedestal = float(np.median(values))
    mad = float(np.median(np.abs(values - pedestal)))
    sigma_est = 1.4826 * mad if mad > 0 else float(np.std(values))
    threshold = max(pedestal + 5.0 * sigma_est, pedestal + 40.0, 150.0)

    signal = values[values >= threshold]
    if signal.size < 30:
        return pedestal, threshold, math.nan

    upper = float(np.percentile(signal, 99.5))
    if upper <= threshold:
        upper = float(signal.max())

    bins = int(np.clip(np.sqrt(signal.size), 30, 140))
    hist, edges = np.histogram(signal, bins=bins, range=(threshold, upper))
    peak_bin = int(np.argmax(hist))
    mip_peak = float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)
    return pedestal, threshold, mip_peak


def build_channel_metrics(list_data: ListData) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total_samples = sum(arr.size for arr in list_data.channel_hg.values())

    for channel in range(64):
        hg = list_data.channel_hg[channel].astype(np.float64)
        lg = list_data.channel_lg[channel].astype(np.float64)
        pedestal, threshold, mip_peak = estimate_mip_peak(hg)

        records.append(
            {
                "channel": channel,
                "samples": int(hg.size),
                "occupancy_fraction": float(hg.size / total_samples) if total_samples else math.nan,
                "hg_mean": float(np.mean(hg)) if hg.size else math.nan,
                "hg_median": float(np.median(hg)) if hg.size else math.nan,
                "hg_std": float(np.std(hg)) if hg.size else math.nan,
                "hg_p95": float(np.percentile(hg, 95)) if hg.size else math.nan,
                "lg_mean": float(np.mean(lg)) if lg.size else math.nan,
                "lg_median": float(np.median(lg)) if lg.size else math.nan,
                "mip_pedestal_hg": pedestal,
                "mip_threshold_hg": threshold,
                "mip_peak_hg": mip_peak,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("channel").reset_index(drop=True)


def build_rate_dataframe(
    list_data: ListData, rate_bin_sec: int, tz_name: str
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    event_ts = pd.to_datetime(list_data.run_start_utc) + pd.to_timedelta(list_data.event_t_us, unit="us")
    event_df = pd.DataFrame(
        {
            "timestamp_utc": event_ts,
            "nhits": list_data.event_nhits.astype(np.int32),
            "event_count": np.ones_like(list_data.event_nhits, dtype=np.int16),
        }
    )

    rate_df = (
        event_df.set_index("timestamp_utc")
        .resample(f"{rate_bin_sec}s")
        .agg(events=("event_count", "sum"), integrated_hits=("nhits", "sum"))
    )
    rate_df["trigger_rate_hz"] = rate_df["events"] / rate_bin_sec
    rate_df["integrated_hit_rate_hz"] = rate_df["integrated_hits"] / rate_bin_sec
    rate_df = rate_df.reset_index()
    rate_df["timestamp_la"] = rate_df["timestamp_utc"].dt.tz_convert(tz_name)

    return rate_df, event_df["timestamp_utc"].min(), event_df["timestamp_utc"].max()


def plot_adc_histograms_by_channel(
    channel_metrics: pd.DataFrame, list_data: ListData, output_path: Path
) -> None:
    all_hg = np.concatenate([list_data.channel_hg[ch] for ch in range(64)])
    x_max = float(np.percentile(all_hg, 99.8))
    x_max = max(300.0, min(8192.0, x_max))

    fig, axes = plt.subplots(8, 8, figsize=(20, 18), constrained_layout=True, sharex=True, sharey=True)
    bins = np.linspace(0, x_max, 80)

    mip_lookup = channel_metrics.set_index("channel")["mip_peak_hg"].to_dict()

    for ch in range(64):
        ax = axes[ch // 8, ch % 8]
        hg = list_data.channel_hg[ch]
        ax.hist(hg, bins=bins, color="#1f77b4", alpha=0.8)
        mip = mip_lookup.get(ch, math.nan)
        if not np.isnan(mip):
            ax.axvline(mip, color="#d62728", lw=1.0)
        ax.set_title(f"Ch {ch:02d}", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

    fig.suptitle("HG ADC Distribution by Channel (red line: estimated MIP peak)", fontsize=15)
    fig.supxlabel("HG ADC")
    fig.supylabel("Counts")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mip_peaks(channel_metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        channel_metrics["channel"],
        channel_metrics["mip_peak_hg"],
        marker="o",
        lw=1.2,
        ms=4,
        color="#d62728",
        label="Estimated MIP peak (HG)",
    )
    ax.plot(
        channel_metrics["channel"],
        channel_metrics["mip_pedestal_hg"],
        marker=".",
        lw=1.0,
        ms=6,
        color="#1f77b4",
        label="Pedestal estimate (HG)",
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("ADC")
    ax.set_title("Derived MIP Peak and Pedestal per Channel")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rate_timeseries(rate_df: pd.DataFrame, output_utc: Path, output_la: Path) -> None:
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(rate_df["timestamp_utc"], rate_df["trigger_rate_hz"], color="#1f77b4", lw=1.2, label="Trigger rate [Hz]")
    ax1.plot(
        rate_df["timestamp_utc"],
        rate_df["integrated_hit_rate_hz"],
        color="#ff7f0e",
        lw=1.1,
        label="Integrated channel rate [Hz]",
    )
    ax1.set_title("Run Rate Time Series (UTC)")
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Rate [Hz]")
    ax1.grid(True, alpha=0.25)
    ax1.legend()
    fig1.autofmt_xdate()
    fig1.savefig(output_utc, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.plot(rate_df["timestamp_la"], rate_df["trigger_rate_hz"], color="#1f77b4", lw=1.2, label="Trigger rate [Hz]")
    ax2.plot(
        rate_df["timestamp_la"],
        rate_df["integrated_hit_rate_hz"],
        color="#ff7f0e",
        lw=1.1,
        label="Integrated channel rate [Hz]",
    )
    ax2.set_title("Run Rate Time Series (America/Los_Angeles)")
    ax2.set_xlabel("Time (America/Los_Angeles)")
    ax2.set_ylabel("Rate [Hz]")
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    fig2.autofmt_xdate()
    fig2.savefig(output_la, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def plot_service_info(service_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)

    for column, color in [
        ("BrdTemp", "#1f77b4"),
        ("DetTemp", "#2ca02c"),
        ("FPGATemp", "#d62728"),
        ("HVTemp", "#9467bd"),
    ]:
        axes[0].plot(service_df["timestamp_utc"], service_df[column], lw=1.0, label=column, color=color)
    axes[0].set_title("Board Environment Metrics (UTC)")
    axes[0].set_ylabel("Temperature [C]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(service_df["timestamp_utc"], service_df["Vmon"], lw=1.1, label="Vmon [V]", color="#ff7f0e")
    axes[1].plot(service_df["timestamp_utc"], service_df["Imon"], lw=1.1, label="Imon [mA]", color="#8c564b")
    axes[1].set_title("HV Monitoring (UTC)")
    axes[1].set_ylabel("Monitor Value")
    axes[1].set_xlabel("Time (UTC)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_dashboard(
    dashboard_path: Path,
    summary_metrics: dict[str, Any],
    metadata: dict[str, Any],
    plot_paths: dict[str, str],
) -> None:
    cfg = metadata.get("config", {})
    rows = []
    for key in [
        "AcquisitionMode",
        "TriggerLogic",
        "HG_Gain",
        "LG_Gain",
        "Pedestal",
        "HV_Vbias",
        "TD_CoarseThreshold",
        "QD_CoarseThreshold",
    ]:
        rows.append(f"<tr><td>{key}</td><td>{cfg.get(key, 'N/A')}</td></tr>")

    summary_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary_metrics.items() if k != "run_start_utc"
    )
    config_rows = "".join(rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QA Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
    h1, h2 {{ margin-bottom: 0.3rem; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    img {{ width: 100%; max-width: 1200px; border: 1px solid #ddd; }}
    .meta {{ margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>Run QA Dashboard</h1>
  <p class="meta"><strong>Run:</strong> {metadata.get('run_number', 'N/A')} | <strong>UTC start:</strong> {summary_metrics.get('run_start_utc', 'N/A')}</p>

  <h2>Summary Metrics</h2>
  <table>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Selected Metadata</h2>
  <table>
    <tbody>{config_rows}</tbody>
  </table>

  <div class="grid">
    <section>
      <h2>ADC per Channel (HG)</h2>
      <img src="{plot_paths['adc_by_channel']}" alt="ADC by channel">
    </section>
    <section>
      <h2>Derived MIP Peaks</h2>
      <img src="{plot_paths['mip_peaks']}" alt="MIP peaks">
    </section>
    <section>
      <h2>Rate Time Series (UTC)</h2>
      <img src="{plot_paths['rate_utc']}" alt="Rate UTC">
    </section>
    <section>
      <h2>Rate Time Series (America/Los_Angeles)</h2>
      <img src="{plot_paths['rate_la']}" alt="Rate LA">
    </section>
    <section>
      <h2>Service Monitoring</h2>
      <img src="{plot_paths['service']}" alt="Service monitoring">
    </section>
  </div>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")


def make_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA metrics, plots, and dashboard from Janus Run files.")
    parser.add_argument("--list-file", type=Path, default=Path("data/raw/Run7_list.txt"))
    parser.add_argument("--info-file", type=Path, default=Path("data/raw/Run7_Info.txt"))
    parser.add_argument("--service-file", type=Path, default=Path("data/raw/Run7_ServiceInfo.txt"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--timezone", type=str, default="America/Los_Angeles")
    parser.add_argument("--rate-bin-sec", type=int, default=60)
    args = parser.parse_args()

    plots_dir = args.outdir / "plots"
    tables_dir = args.outdir / "tables"
    dashboard_dir = args.outdir / "dashboard"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    metadata = parse_run_info(args.info_file)
    list_data = parse_list_file(args.list_file)
    service_df = parse_service_info(args.service_file)
    channel_metrics = build_channel_metrics(list_data)
    rate_df, run_start_evt, run_stop_evt = build_rate_dataframe(list_data, args.rate_bin_sec, args.timezone)

    plot_adc_histograms_by_channel(channel_metrics, list_data, plots_dir / "adc_hg_by_channel.png")
    plot_mip_peaks(channel_metrics, plots_dir / "mip_peak_by_channel.png")
    plot_rate_timeseries(
        rate_df,
        plots_dir / "rate_timeseries_utc.png",
        plots_dir / "rate_timeseries_los_angeles.png",
    )
    plot_service_info(service_df, plots_dir / "service_monitoring_utc.png")

    channel_metrics.to_csv(tables_dir / "channel_metrics.csv", index=False)
    rate_df.to_csv(tables_dir / "rate_timeseries.csv", index=False)
    service_df.to_csv(tables_dir / "service_info_parsed.csv", index=False)

    valid_mip = channel_metrics["mip_peak_hg"].dropna()
    run_duration_s = (run_stop_evt - run_start_evt).total_seconds()
    summary_metrics: dict[str, Any] = {
        "run_start_utc": list_data.run_start_utc.isoformat(),
        "run_start_event_utc": run_start_evt.isoformat(),
        "run_stop_event_utc": run_stop_evt.isoformat(),
        "events": int(list_data.event_t_us.size),
        "channels": 64,
        "total_samples": int(sum(arr.size for arr in list_data.channel_hg.values())),
        "run_duration_s_from_events": float(run_duration_s),
        "trigger_rate_hz_avg": float(list_data.event_t_us.size / run_duration_s) if run_duration_s > 0 else math.nan,
        "integrated_hit_rate_hz_avg": float(np.sum(list_data.event_nhits) / run_duration_s) if run_duration_s > 0 else math.nan,
        "mip_peak_channels_with_estimate": int(valid_mip.size),
        "mip_peak_hg_mean_adc": float(valid_mip.mean()) if valid_mip.size else math.nan,
        "mip_peak_hg_std_adc": float(valid_mip.std()) if valid_mip.size else math.nan,
        "vmon_mean": float(service_df["Vmon"].mean()),
        "imon_mean_ma": float(service_df["Imon"].mean()),
        "board_temp_mean_c": float(service_df["BrdTemp"].mean()),
        "det_temp_mean_c": float(service_df["DetTemp"].mean()),
        "fpga_temp_mean_c": float(service_df["FPGATemp"].mean()),
        "hv_temp_mean_c": float(service_df["HVTemp"].mean()),
    }

    (tables_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=make_serializable), encoding="utf-8"
    )
    (tables_dir / "summary_metrics.json").write_text(
        json.dumps({k: make_serializable(v) for k, v in summary_metrics.items()}, indent=2), encoding="utf-8"
    )

    write_dashboard(
        dashboard_path=dashboard_dir / "index.html",
        summary_metrics={k: make_serializable(v) for k, v in summary_metrics.items()},
        metadata=metadata,
        plot_paths={
            "adc_by_channel": "../plots/adc_hg_by_channel.png",
            "mip_peaks": "../plots/mip_peak_by_channel.png",
            "rate_utc": "../plots/rate_timeseries_utc.png",
            "rate_la": "../plots/rate_timeseries_los_angeles.png",
            "service": "../plots/service_monitoring_utc.png",
        },
    )

    print("QA report generated.")
    print(f"- Dashboard: {dashboard_dir / 'index.html'}")
    print(f"- Summary metrics: {tables_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
