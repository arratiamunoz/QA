from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
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
    acquisition_mode_header: str | None
    file_format_version: str | None
    board_model: str | None


RUN_START_RE = re.compile(r"Run start time:\s*(.+?)\s*UTC")
RUN_NUM_RE = re.compile(r"Run n\.\s*(\d+)")
START_TIME_RE = re.compile(r"Start Time:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})")
STOP_TIME_RE = re.compile(r"Stop Time:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})")
ELAPSED_RE = re.compile(r"Elapsed time\s*=\s*([\d.]+)\s*s")
SOFTWARE_RE = re.compile(r"Software Version:\s*(.+)")
OUTPUT_FORMAT_RE = re.compile(r"Output data format version:\s*(.+)")
KV_RE = re.compile(r"^([A-Za-z0-9_\[\]]+)\s{2,}(.+?)\s*$")

LIST_MODE_RE = re.compile(r"Acquisition Mode:\s*(.+)")
LIST_FORMAT_RE = re.compile(r"File Format Version\s+(.+)")
LIST_BOARD_RE = re.compile(r"Board:\s*(.+)")

SUPPORTED_MODES = {"SPECTROSCOPY"}


def round_float(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def standardize_mode(raw_mode: str | None) -> str | None:
    if raw_mode is None:
        return None
    normalized = raw_mode.strip().upper().replace("-", "_")
    if normalized in {"SPECTROSCOPY", "SPECT"}:
        return "SPECTROSCOPY"
    if normalized in {"SPECT_TIMING", "SPECTTIMING"}:
        return "SPECT_TIMING"
    if normalized in {"TIMING_CSTART", "TIMING_START"}:
        return "TIMING_CSTART"
    if normalized in {"TIMING_CSTOP", "TIMING_STOP"}:
        return "TIMING_CSTOP"
    if normalized == "COUNTING":
        return "COUNTING"
    if normalized == "WAVEFORM":
        return "WAVEFORM"
    return normalized


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

    software_match = SOFTWARE_RE.search(text)
    if software_match:
        metadata["software_version"] = software_match.group(1).strip()

    output_fmt_match = OUTPUT_FORMAT_RE.search(text)
    if output_fmt_match:
        metadata["output_data_format_version"] = output_fmt_match.group(1).strip()

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        payload = line.split("#", 1)[0].rstrip()
        if not payload:
            continue
        match = KV_RE.match(payload)
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
    acquisition_mode_header: str | None = None
    file_format_version: str | None = None
    board_model: str | None = None

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

                mode_match = LIST_MODE_RE.search(stripped)
                if mode_match:
                    acquisition_mode_header = standardize_mode(mode_match.group(1))

                fmt_match = LIST_FORMAT_RE.search(stripped)
                if fmt_match:
                    file_format_version = fmt_match.group(1).strip()

                board_match = LIST_BOARD_RE.search(stripped)
                if board_match:
                    board_model = board_match.group(1).strip()
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
        acquisition_mode_header=acquisition_mode_header,
        file_format_version=file_format_version,
        board_model=board_model,
    )


def parse_service_info(service_path: Path) -> pd.DataFrame:
    service_df = pd.read_csv(service_path, sep=r"\s+", engine="python")
    service_df["timestamp_utc"] = pd.to_datetime(service_df["TStampPC"], unit="ms", utc=True)
    return service_df


def detect_mode_and_plot_profile(metadata: dict[str, Any], list_data: ListData) -> dict[str, Any]:
    config_mode = standardize_mode(metadata.get("config", {}).get("AcquisitionMode"))
    list_mode = standardize_mode(list_data.acquisition_mode_header)
    detected_mode = config_mode or list_mode or "UNKNOWN"

    mode_warnings: list[str] = []
    if config_mode and list_mode and config_mode != list_mode:
        mode_warnings.append(
            f"Metadata/list mismatch: config mode={config_mode}, list header mode={list_mode}. Using {detected_mode}."
        )

    gain_select = metadata.get("config", {}).get("GainSelect", "BOTH").strip().upper()
    has_hg = gain_select in {"BOTH", "HIGH", "AUTO"}
    has_lg = gain_select in {"BOTH", "LOW"}
    if gain_select not in {"BOTH", "HIGH", "LOW", "AUTO"}:
        mode_warnings.append(f"Unknown GainSelect={gain_select}; defaulting to BOTH behavior.")
        has_hg = True
        has_lg = True

    expected_plots: list[str] = [
        "rate_timeseries_utc",
        "rate_timeseries_los_angeles",
        "service_monitoring_utc",
    ]

    if detected_mode == "SPECTROSCOPY":
        if has_hg:
            expected_plots.append("adc_hg_by_channel")
        if has_lg:
            expected_plots.append("adc_lg_by_channel")
        expected_plots.append("mip_peak_by_channel_hg_lg")
        expected_plots.append("channel_threshold_counts")
    else:
        mode_warnings.append(
            f"Mode {detected_mode} is not yet fully implemented; only generic rate/service plots will be generated."
        )

    return {
        "detected_mode": detected_mode,
        "mode_supported": detected_mode in SUPPORTED_MODES,
        "gain_select": gain_select,
        "expect_hg": has_hg,
        "expect_lg": has_lg,
        "expected_plots": expected_plots,
        "warnings": mode_warnings,
    }


def estimate_mip_peak(
    adc_values: np.ndarray, *, min_signal_offset: float = 40.0, abs_floor: float = 150.0
) -> tuple[float, float, float]:
    values = adc_values.astype(np.float64)
    if values.size < 100:
        return math.nan, math.nan, math.nan

    pedestal = float(np.median(values))
    mad = float(np.median(np.abs(values - pedestal)))
    sigma_est = 1.4826 * mad if mad > 0 else float(np.std(values))
    threshold = max(pedestal + 5.0 * sigma_est, pedestal + min_signal_offset, abs_floor)

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
        hg_pedestal, hg_threshold, hg_mip_peak = estimate_mip_peak(
            hg, min_signal_offset=40.0, abs_floor=150.0
        )
        lg_pedestal, lg_threshold, lg_mip_peak = estimate_mip_peak(
            lg, min_signal_offset=30.0, abs_floor=120.0
        )

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
                "lg_std": float(np.std(lg)) if lg.size else math.nan,
                "lg_p95": float(np.percentile(lg, 95)) if lg.size else math.nan,
                "mip_pedestal_hg": hg_pedestal,
                "mip_threshold_hg": hg_threshold,
                "mip_peak_hg": hg_mip_peak,
                "mip_pedestal_lg": lg_pedestal,
                "mip_threshold_lg": lg_threshold,
                "mip_peak_lg": lg_mip_peak,
            }
        )

    metrics_df = pd.DataFrame.from_records(records).sort_values("channel").reset_index(drop=True)
    float_cols = metrics_df.select_dtypes(include=["float64", "float32"]).columns
    metrics_df[float_cols] = metrics_df[float_cols].round(3)
    return metrics_df


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
        .agg(events=("event_count", "sum"))
    )
    rate_df["trigger_rate_hz"] = rate_df["events"] / rate_bin_sec
    rate_df["trigger_rate_sigma_hz"] = np.sqrt(rate_df["events"]) / rate_bin_sec
    rate_df = rate_df.reset_index()
    rate_df["timestamp_local"] = rate_df["timestamp_utc"].dt.tz_convert(tz_name)

    float_cols = rate_df.select_dtypes(include=["float64", "float32"]).columns
    rate_df[float_cols] = rate_df[float_cols].round(4)

    return rate_df, event_df["timestamp_utc"].min(), event_df["timestamp_utc"].max()


def configure_datetime_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def build_run_time_labels(
    metadata: dict[str, Any], run_start_evt: pd.Timestamp, run_stop_evt: pd.Timestamp, tz_name: str
) -> dict[str, str]:
    run_number = metadata.get("run_number", "N/A")
    mode = metadata.get("detected_mode", "N/A")
    utc_label = f"{run_start_evt:%Y-%m-%d %H:%M:%S} to {run_stop_evt:%Y-%m-%d %H:%M:%S} UTC"

    local_start = run_start_evt.tz_convert(tz_name)
    local_stop = run_stop_evt.tz_convert(tz_name)
    local_label = f"{local_start:%Y-%m-%d %H:%M:%S} to {local_stop:%Y-%m-%d %H:%M:%S} {tz_name}"

    return {
        "header": f"Run {run_number} | Mode {mode}",
        "utc_window": utc_label,
        "local_window": local_label,
    }


def build_plot_context(metadata: dict[str, Any], channel_threshold_adc: int) -> dict[str, str]:
    cfg = metadata.get("config", {})
    trigger_logic = cfg.get("TriggerLogic", "N/A")
    td_thr = cfg.get("TD_CoarseThreshold", "N/A")
    qd_thr = cfg.get("QD_CoarseThreshold", "N/A")
    return {
        "trigger_logic": str(trigger_logic),
        "td_threshold": str(td_thr),
        "qd_threshold": str(qd_thr),
        "channel_threshold_adc": str(channel_threshold_adc),
    }


def build_channel_threshold_summary(list_data: ListData, threshold_adc: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total_events = int(list_data.event_t_us.size)
    for ch in range(64):
        hg = list_data.channel_hg[ch]
        lg = list_data.channel_lg[ch]
        hg_count = int(np.sum(hg >= threshold_adc))
        lg_count = int(np.sum(lg >= threshold_adc))
        records.append(
            {
                "channel": ch,
                "threshold_adc": threshold_adc,
                "hg_count_above_threshold": hg_count,
                "hg_fraction_above_threshold": (hg_count / total_events) if total_events else math.nan,
                "lg_count_above_threshold": lg_count,
                "lg_fraction_above_threshold": (lg_count / total_events) if total_events else math.nan,
            }
        )

    df = pd.DataFrame.from_records(records).sort_values("channel").reset_index(drop=True)
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].round(4)
    return df


def plot_adc_histograms_by_channel(
    channel_data: dict[int, np.ndarray],
    channel_metrics: pd.DataFrame,
    gain_label: str,
    output_path: Path,
    run_labels: dict[str, str],
    plot_context: dict[str, str],
) -> None:
    all_values = np.concatenate([channel_data[ch] for ch in range(64)])
    x_max = float(np.percentile(all_values, 99.8))
    x_max = max(220.0, min(8192.0, x_max))

    fig, axes = plt.subplots(8, 8, figsize=(24, 22), constrained_layout=True, sharex=True, sharey=True)
    bins = np.linspace(0, x_max, 82)

    peak_col = f"mip_peak_{gain_label.lower()}"
    mip_lookup = channel_metrics.set_index("channel")[peak_col].to_dict()
    hist_color = "#1f77b4" if gain_label.upper() == "HG" else "#2ca02c"

    for ch in range(64):
        ax = axes[ch // 8, ch % 8]
        values = channel_data[ch]
        ax.hist(values, bins=bins, color=hist_color, alpha=0.85, edgecolor="none")
        mip = mip_lookup.get(ch, math.nan)
        if not np.isnan(mip):
            ax.axvline(mip, color="#d62728", lw=1.2, alpha=0.95)
        ax.set_title(f"Ch {ch:02d}", fontsize=11)
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        f"{gain_label.upper()} ADC Distribution by Channel (red = estimated MIP peak)\n"
        f"{run_labels['header']} | {run_labels['utc_window']}",
        fontsize=18,
    )
    fig.supxlabel(f"{gain_label.upper()} ADC", fontsize=14)
    fig.supylabel("Counts", fontsize=14)
    fig.text(
        0.5,
        0.012,
        f"TriggerLogic={plot_context['trigger_logic']} | TD={plot_context['td_threshold']} | "
        f"QD={plot_context['qd_threshold']} | Local: {run_labels['local_window']}",
        ha="center",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mip_peaks(
    channel_metrics: pd.DataFrame, output_path: Path, run_labels: dict[str, str], plot_context: dict[str, str]
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.plot(
        channel_metrics["channel"],
        channel_metrics["mip_peak_hg"],
        marker="o",
        lw=2.0,
        ms=4.5,
        color="#d62728",
        label="MIP peak HG",
    )
    ax.plot(
        channel_metrics["channel"],
        channel_metrics["mip_peak_lg"],
        marker="s",
        lw=2.0,
        ms=4.0,
        color="#2ca02c",
        label="MIP peak LG",
    )
    ax.plot(
        channel_metrics["channel"],
        channel_metrics["mip_pedestal_hg"],
        lw=1.5,
        ls="--",
        color="#1f77b4",
        alpha=0.85,
        label="Pedestal HG",
    )
    ax.plot(
        channel_metrics["channel"],
        channel_metrics["mip_pedestal_lg"],
        lw=1.5,
        ls="--",
        color="#17becf",
        alpha=0.85,
        label="Pedestal LG",
    )
    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel("ADC", fontsize=12)
    ax.set_title("Derived MIP Peak and Pedestal per Channel", fontsize=14)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=10.5, ncol=2, framealpha=0.95)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=16))
    ax.tick_params(labelsize=10)
    ax.text(
        0.985,
        0.04,
        f"{run_labels['header']}\nUTC: {run_labels['utc_window']}\n"
        f"TriggerLogic={plot_context['trigger_logic']}, TD={plot_context['td_threshold']}, "
        f"QD={plot_context['qd_threshold']}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88, "edgecolor": "#cccccc"},
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_rate_single_axis(
    ax: plt.Axes,
    x_values: pd.Series,
    rate_df: pd.DataFrame,
    x_label: str,
    run_labels: dict[str, str],
    zone_label: str,
    plot_context: dict[str, str],
) -> None:
    trig = rate_df["trigger_rate_hz"].to_numpy(dtype=float)
    trig_sigma = rate_df["trigger_rate_sigma_hz"].to_numpy(dtype=float)

    ax.plot(x_values, trig, color="#1f77b4", lw=2.5, label="Trigger rate [Hz]")

    ax.fill_between(x_values, trig - trig_sigma, trig + trig_sigma, color="#1f77b4", alpha=0.12)

    step = max(1, len(rate_df) // 80)
    reduced = rate_df.iloc[::step]
    reduced_x = x_values.iloc[::step]
    ax.errorbar(
        reduced_x,
        reduced["trigger_rate_hz"],
        yerr=reduced["trigger_rate_sigma_hz"],
        fmt="none",
        ecolor="#1f77b4",
        alpha=0.35,
        elinewidth=0.9,
        capsize=1.8,
    )

    ax.set_title(f"Run Rate Time Series ({zone_label})", fontsize=14)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Rate [Hz]", fontsize=11)
    ax.grid(True, alpha=0.26)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    configure_datetime_axis(ax)
    ax.text(
        0.01,
        0.98,
        f"{run_labels['header']}\nUTC: {run_labels['utc_window']}\n"
        f"TriggerLogic={plot_context['trigger_logic']} | "
        f"TD={plot_context['td_threshold']} | QD={plot_context['qd_threshold']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )


def plot_rate_timeseries(
    rate_df: pd.DataFrame,
    output_utc: Path,
    output_local: Path,
    run_labels: dict[str, str],
    tz_name: str,
    plot_context: dict[str, str],
) -> None:
    fig1, ax1 = plt.subplots(figsize=(14, 5.8))
    plot_rate_single_axis(
        ax1,
        rate_df["timestamp_utc"],
        rate_df,
        "Time (UTC)",
        run_labels=run_labels,
        zone_label="UTC",
        plot_context=plot_context,
    )
    fig1.savefig(output_utc, dpi=165, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 5.8))
    plot_rate_single_axis(
        ax2,
        rate_df["timestamp_local"],
        rate_df,
        f"Time ({tz_name})",
        run_labels=run_labels,
        zone_label=tz_name,
        plot_context=plot_context,
    )
    fig2.savefig(output_local, dpi=165, bbox_inches="tight")
    plt.close(fig2)


def plot_channel_threshold_counts(
    threshold_df: pd.DataFrame,
    output_path: Path,
    run_labels: dict[str, str],
    plot_context: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.2))
    channels = threshold_df["channel"].to_numpy(dtype=float)
    hg_counts = threshold_df["hg_count_above_threshold"].to_numpy(dtype=float)
    lg_counts = threshold_df["lg_count_above_threshold"].to_numpy(dtype=float)
    threshold_adc = int(threshold_df["threshold_adc"].iloc[0])

    ax.plot(channels, hg_counts, marker="o", lw=2.2, ms=4.5, color="#1f77b4", label="HG count")
    ax.plot(channels, lg_counts, marker="s", lw=2.2, ms=4.2, color="#2ca02c", label="LG count")
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel("Events above threshold", fontsize=12)
    ax.set_title(f"Channel Occupancy Above {threshold_adc} ADC (dead-channel screening)", fontsize=14)
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(loc="upper right", fontsize=10.5, framealpha=0.95)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=16))
    ax.tick_params(labelsize=10)

    dead_hg = int(np.sum(hg_counts == 0))
    dead_lg = int(np.sum(lg_counts == 0))
    dead_hg_channels = channels[hg_counts == 0]
    dead_lg_channels = channels[lg_counts == 0]
    if dead_hg_channels.size:
        ax.scatter(dead_hg_channels, np.zeros_like(dead_hg_channels), color="#1f77b4", marker="x", s=30, alpha=0.9)
    if dead_lg_channels.size:
        ax.scatter(dead_lg_channels, np.zeros_like(dead_lg_channels), color="#2ca02c", marker="x", s=30, alpha=0.9)
    ax.text(
        0.01,
        0.98,
        f"{run_labels['header']}\nUTC: {run_labels['utc_window']}\n"
        f"Threshold={threshold_adc} ADC | TriggerLogic={plot_context['trigger_logic']} | "
        f"TD={plot_context['td_threshold']} | QD={plot_context['qd_threshold']}\n"
        f"Channels with zero count: HG={dead_hg}, LG={dead_lg}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_service_info(service_df: pd.DataFrame, output_path: Path, run_labels: dict[str, str]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.6), sharex=True, constrained_layout=True)

    for column, color in [
        ("BrdTemp", "#1f77b4"),
        ("DetTemp", "#2ca02c"),
        ("FPGATemp", "#d62728"),
        ("HVTemp", "#9467bd"),
    ]:
        axes[0].plot(service_df["timestamp_utc"], service_df[column], lw=1.8, label=column, color=color)
    axes[0].set_title("Board Environment Metrics (UTC)", fontsize=13)
    axes[0].set_ylabel("Temperature [C]", fontsize=11)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", ncol=4, fontsize=9.5, framealpha=0.95)

    axes[1].plot(service_df["timestamp_utc"], service_df["Vmon"], lw=1.9, label="Vmon [V]", color="#ff7f0e")
    axes[1].plot(service_df["timestamp_utc"], service_df["Imon"], lw=1.9, label="Imon [mA]", color="#8c564b")
    axes[1].set_title("HV Monitoring (UTC)", fontsize=13)
    axes[1].set_ylabel("Monitor Value", fontsize=11)
    axes[1].set_xlabel("Time (UTC)", fontsize=11)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left", fontsize=10, framealpha=0.95)

    configure_datetime_axis(axes[1])
    axes[0].text(
        0.99,
        0.97,
        f"{run_labels['header']}\n{run_labels['utc_window']}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9.1,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )

    fig.savefig(output_path, dpi=165, bbox_inches="tight")
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
        "GainSelect",
        "TriggerLogic",
        "HG_Gain",
        "LG_Gain",
        "Pedestal",
        "HV_Vbias",
        "TD_CoarseThreshold",
        "QD_CoarseThreshold",
    ]:
        rows.append(f"<tr><td>{key}</td><td>{cfg.get(key, 'N/A')}</td></tr>")

    summary_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary_metrics.items())
    config_rows = "".join(rows)

    display_order = [
        ("adc_hg_by_channel", "ADC per Channel (HG)"),
        ("adc_lg_by_channel", "ADC per Channel (LG)"),
        ("mip_peak_by_channel_hg_lg", "Derived MIP Peaks (HG + LG)"),
        ("channel_threshold_counts", "Counts Above ADC Threshold"),
        ("rate_timeseries_utc", "Rate Time Series (UTC)"),
        ("rate_timeseries_los_angeles", "Rate Time Series (America/Los_Angeles)"),
        ("service_monitoring_utc", "Service Monitoring (UTC)"),
    ]

    sections = []
    for key, title in display_order:
        if key in plot_paths:
            sections.append(
                f"<section><h2>{title}</h2><img src=\"{plot_paths[key]}\" alt=\"{title}\"></section>"
            )

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
  <p class="meta"><strong>Run:</strong> {metadata.get('run_number', 'N/A')} | <strong>Detected mode:</strong> {metadata.get('detected_mode', 'N/A')}</p>

  <h2>Summary Metrics</h2>
  <table>
    <tbody>{summary_rows}</tbody>
  </table>

  <h2>Selected Metadata</h2>
  <table>
    <tbody>{config_rows}</tbody>
  </table>

  <div class="grid">
    {"".join(sections)}
  </div>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")


def make_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QA metrics, plots, and dashboard from Janus run list/info/service files."
    )
    parser.add_argument("--list-file", type=Path, default=Path("data/raw/Run7_list.txt"))
    parser.add_argument("--info-file", type=Path, default=Path("data/raw/Run7_Info.txt"))
    parser.add_argument("--service-file", type=Path, default=Path("data/raw/Run7_ServiceInfo.txt"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--timezone", type=str, default="America/Los_Angeles")
    parser.add_argument("--rate-bin-sec", type=int, default=60)
    parser.add_argument("--channel-threshold-adc", type=int, default=200)
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    plots_dir = args.outdir / "plots"
    tables_dir = args.outdir / "tables"
    dashboard_dir = args.outdir / "dashboard"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    metadata = parse_run_info(args.info_file)
    list_data = parse_list_file(args.list_file)
    service_df = parse_service_info(args.service_file)

    mode_profile = detect_mode_and_plot_profile(metadata, list_data)
    metadata["detected_mode"] = mode_profile["detected_mode"]
    metadata["mode_profile"] = mode_profile
    metadata["list_file_format_version"] = list_data.file_format_version
    metadata["list_header_board"] = list_data.board_model

    channel_metrics = build_channel_metrics(list_data)
    threshold_summary_df = build_channel_threshold_summary(list_data, threshold_adc=args.channel_threshold_adc)
    rate_df, run_start_evt, run_stop_evt = build_rate_dataframe(list_data, args.rate_bin_sec, args.timezone)
    run_labels = build_run_time_labels(metadata, run_start_evt, run_stop_evt, args.timezone)
    plot_context = build_plot_context(metadata, channel_threshold_adc=args.channel_threshold_adc)

    generated_plots: dict[str, str] = {}

    if "adc_hg_by_channel" in mode_profile["expected_plots"]:
        plot_adc_histograms_by_channel(
            channel_data=list_data.channel_hg,
            channel_metrics=channel_metrics,
            gain_label="HG",
            output_path=plots_dir / "adc_hg_by_channel.png",
            run_labels=run_labels,
            plot_context=plot_context,
        )
        generated_plots["adc_hg_by_channel"] = "../plots/adc_hg_by_channel.png"

    if "adc_lg_by_channel" in mode_profile["expected_plots"]:
        plot_adc_histograms_by_channel(
            channel_data=list_data.channel_lg,
            channel_metrics=channel_metrics,
            gain_label="LG",
            output_path=plots_dir / "adc_lg_by_channel.png",
            run_labels=run_labels,
            plot_context=plot_context,
        )
        generated_plots["adc_lg_by_channel"] = "../plots/adc_lg_by_channel.png"

    if "mip_peak_by_channel_hg_lg" in mode_profile["expected_plots"]:
        plot_mip_peaks(
            channel_metrics,
            plots_dir / "mip_peak_by_channel_hg_lg.png",
            run_labels,
            plot_context=plot_context,
        )
        generated_plots["mip_peak_by_channel_hg_lg"] = "../plots/mip_peak_by_channel_hg_lg.png"

    if "channel_threshold_counts" in mode_profile["expected_plots"]:
        plot_channel_threshold_counts(
            threshold_df=threshold_summary_df,
            output_path=plots_dir / "channel_threshold_counts.png",
            run_labels=run_labels,
            plot_context=plot_context,
        )
        generated_plots["channel_threshold_counts"] = "../plots/channel_threshold_counts.png"

    if "rate_timeseries_utc" in mode_profile["expected_plots"] or "rate_timeseries_los_angeles" in mode_profile[
        "expected_plots"
    ]:
        plot_rate_timeseries(
            rate_df,
            plots_dir / "rate_timeseries_utc.png",
            plots_dir / "rate_timeseries_los_angeles.png",
            run_labels=run_labels,
            tz_name=args.timezone,
            plot_context=plot_context,
        )
        generated_plots["rate_timeseries_utc"] = "../plots/rate_timeseries_utc.png"
        generated_plots["rate_timeseries_los_angeles"] = "../plots/rate_timeseries_los_angeles.png"

    if "service_monitoring_utc" in mode_profile["expected_plots"]:
        plot_service_info(service_df, plots_dir / "service_monitoring_utc.png", run_labels)
        generated_plots["service_monitoring_utc"] = "../plots/service_monitoring_utc.png"

    channel_metrics.to_csv(tables_dir / "channel_metrics.csv", index=False)
    threshold_summary_df.to_csv(tables_dir / "channel_threshold_summary.csv", index=False)
    rate_df.to_csv(tables_dir / "rate_timeseries.csv", index=False)
    service_df.to_csv(tables_dir / "service_info_parsed.csv", index=False)

    valid_mip_hg = channel_metrics["mip_peak_hg"].dropna()
    valid_mip_lg = channel_metrics["mip_peak_lg"].dropna()
    run_duration_s = (run_stop_evt - run_start_evt).total_seconds()
    summary_metrics: dict[str, Any] = {
        "run_start_utc": list_data.run_start_utc.isoformat(),
        "run_start_event_utc": run_start_evt.isoformat(),
        "run_stop_event_utc": run_stop_evt.isoformat(),
        "run_start_local": run_start_evt.tz_convert(args.timezone).isoformat(),
        "run_stop_local": run_stop_evt.tz_convert(args.timezone).isoformat(),
        "detected_mode": mode_profile["detected_mode"],
        "gain_select": mode_profile["gain_select"],
        "list_file_format_version": list_data.file_format_version,
        "channel_threshold_adc": int(args.channel_threshold_adc),
        "trigger_logic": metadata.get("config", {}).get("TriggerLogic", "N/A"),
        "td_coarse_threshold": metadata.get("config", {}).get("TD_CoarseThreshold", "N/A"),
        "qd_coarse_threshold": metadata.get("config", {}).get("QD_CoarseThreshold", "N/A"),
        "events": int(list_data.event_t_us.size),
        "channels": 64,
        "total_samples": int(sum(arr.size for arr in list_data.channel_hg.values())),
        "run_duration_s_from_events": round_float(run_duration_s, 3),
        "trigger_rate_hz_avg": round_float(float(list_data.event_t_us.size / run_duration_s), 4)
        if run_duration_s > 0
        else math.nan,
        "mip_peak_hg_channels_with_estimate": int(valid_mip_hg.size),
        "mip_peak_hg_mean_adc": round_float(float(valid_mip_hg.mean()), 3) if valid_mip_hg.size else math.nan,
        "mip_peak_hg_std_adc": round_float(float(valid_mip_hg.std()), 3) if valid_mip_hg.size else math.nan,
        "mip_peak_lg_channels_with_estimate": int(valid_mip_lg.size),
        "mip_peak_lg_mean_adc": round_float(float(valid_mip_lg.mean()), 3) if valid_mip_lg.size else math.nan,
        "mip_peak_lg_std_adc": round_float(float(valid_mip_lg.std()), 3) if valid_mip_lg.size else math.nan,
        "channels_with_zero_hg_above_threshold": int(np.sum(threshold_summary_df["hg_count_above_threshold"] == 0)),
        "channels_with_zero_lg_above_threshold": int(np.sum(threshold_summary_df["lg_count_above_threshold"] == 0)),
        "vmon_mean_v": round_float(float(service_df["Vmon"].mean()), 4),
        "imon_mean_ma": round_float(float(service_df["Imon"].mean()), 5),
        "board_temp_mean_c": round_float(float(service_df["BrdTemp"].mean()), 3),
        "det_temp_mean_c": round_float(float(service_df["DetTemp"].mean()), 3),
        "fpga_temp_mean_c": round_float(float(service_df["FPGATemp"].mean()), 3),
        "hv_temp_mean_c": round_float(float(service_df["HVTemp"].mean()), 3),
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
        plot_paths=generated_plots,
    )

    print("QA report generated.")
    print(f"- Detected mode: {mode_profile['detected_mode']}")
    if mode_profile["warnings"]:
        print(f"- Warnings: {' | '.join(mode_profile['warnings'])}")
    print(f"- Dashboard: {dashboard_dir / 'index.html'}")
    print(f"- Summary metrics: {tables_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
