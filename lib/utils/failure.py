from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import Optional

from ..classes.Satellite import Satellite
from .metrics import run_epidemic_metrics, MetricsConfig


@dataclass
class FailureConfig:
    """Describes a satellite failure scenario.

    Voluntary failures are listed explicitly; random failures are drawn
    from a pool (excluding `exclude_from_random`).  Both types can be
    combined.  The failure window [t_start, t_end] is inclusive;
    t_end=None means "until the last simulation step".
    """

    failed_satellite_names: list[str] = field(default_factory=list)

    n_random_failures: int = 0
    random_seed: Optional[int] = None

    t_start: int = 0
    t_end: Optional[int] = None

    # Satellites never picked by the random selector (e.g. source / dest)
    exclude_from_random: list[str] = field(default_factory=list)


# ── Internal helpers ────────────────────────────────────────────────────────

def _resolve_failed_names(
    satellites: list[Satellite],
    config: FailureConfig,
) -> list[str]:
    failed = list(config.failed_satellite_names)
    if config.n_random_failures > 0:
        rng = random.Random(config.random_seed)
        pool = [
            s.name for s in satellites
            if s.name not in config.exclude_from_random
            and s.name not in failed
        ]
        n = min(config.n_random_failures, len(pool))
        failed.extend(rng.sample(pool, n))
    return failed


# ── Public API ───────────────────────────────────────────────────────────────

def apply_failures(
    satellites: list[Satellite],
    failed_sat_names: list[str],
    t_start: int,
    t_end: int,
) -> list[Satellite]:
    """Return a deep copy of *satellites* with topology failures applied.

    Over [t_start, t_end] each failed satellite is fully isolated:
    it loses all its links, and each former neighbor loses its link back.
    """
    sat_copy = copy.deepcopy(satellites)
    sat_map = {s.name: s for s in sat_copy}

    for name in failed_sat_names:
        if name not in sat_map:
            raise ValueError(f"Unknown satellite '{name}'")
        failed = sat_map[name]
        n_steps = len(failed.list_coordinates)

        for t in range(t_start, min(t_end + 1, n_steps)):
            instant = failed.list_coordinates[t]
            # Snapshot neighbors before the dict is modified
            neighbors_now = list(instant.neighbors)
            # Sever link from each neighbor back to the failed satellite
            for nbr in neighbors_now:
                nbr.list_coordinates[t].deactivate_neighbors([failed.id])
            # Sever all links out of the failed satellite
            instant.deactivate_all_neighbors()

    return sat_copy


def run_failure_comparison(
    satellites: list[Satellite],
    source_name: str,
    dest_name: str,
    failure_config: FailureConfig,
    metrics_config: Optional[MetricsConfig] = None,
    n_pdus: int = 5,
    injection_interval: int = 20,
    max_steps: int = 500,
) -> dict:
    """Run baseline + failure simulations and return a comparison dict."""
    failed_names = _resolve_failed_names(satellites, failure_config)

    t_end = failure_config.t_end
    if t_end is None:
        t_end = len(satellites[0].list_coordinates) - 1

    baseline = run_epidemic_metrics(
        satellites, source_name, dest_name,
        n_pdus=n_pdus,
        injection_interval=injection_interval,
        config=metrics_config,
        max_steps=max_steps,
    )

    broken_sats = apply_failures(
        satellites, failed_names, failure_config.t_start, t_end
    )
    with_failure = run_epidemic_metrics(
        broken_sats, source_name, dest_name,
        n_pdus=n_pdus,
        injection_interval=injection_interval,
        config=metrics_config,
        max_steps=max_steps,
    )

    return {
        'baseline': baseline,
        'failure': with_failure,
        'failed_satellites': failed_names,
        'failure_config': failure_config,
        't_start': failure_config.t_start,
        't_end': t_end,
    }


def print_comparison_table(
    comparison: dict,
    source: str = "",
    dest: str = "",
) -> None:
    """Print a formatted side-by-side comparison of baseline vs failure."""
    bsl = comparison['baseline']
    flt = comparison['failure']
    failed = comparison['failed_satellites']
    t_start = comparison['t_start']
    t_end = comparison['t_end']
    W = 72

    def _fmt(v):
        return "N/A" if v is None else f"{v:.2f}"

    def _delta(b, f):
        if b is None or f is None:
            return "N/A"
        if b == 0:
            return "+inf" if f > 0 else "=0"
        return f"{(f - b) / b * 100:+.1f}%"

    print("=" * W)
    print("  COMPARAISON AVANT / APRES PANNES")
    print("=" * W)
    if source and dest:
        print(f"  Source --> Dest      : {source} --> {dest}")
    sat_list = ', '.join(failed) if failed else "aucun"
    print(f"  Satellites en panne  : {sat_list}")
    panne_type = []
    if comparison['failure_config'].failed_satellite_names:
        panne_type.append("volontaire")
    if comparison['failure_config'].n_random_failures:
        panne_type.append("aleatoire")
    print(f"  Type                 : {' + '.join(panne_type) if panne_type else '-'}")
    print(f"  Fenetre de panne     : pas {t_start} --> {t_end}")
    print("-" * W)
    print(f"  {'Metrique':<30} {'Avant':>10} {'Apres':>10} {'Delta':>10}")
    print("-" * W)

    rows = [
        ("Taux de livraison (%)", bsl['delivery_ratio'] * 100, flt['delivery_ratio'] * 100),
        ("PDUs livres",           bsl['n_delivered'],           flt['n_delivered']),
        ("Goodput (bps)",         bsl['goodput_bps'],           flt['goodput_bps']),
        ("Latence moy. (s)",      bsl.get('mean_latency_s'),    flt.get('mean_latency_s')),
        ("Latence min (s)",       bsl.get('min_latency_s'),     flt.get('min_latency_s')),
        ("Latence max (s)",       bsl.get('max_latency_s'),     flt.get('max_latency_s')),
        ("Ecart-type lat. (s)",   bsl.get('std_latency_s'),     flt.get('std_latency_s')),
        ("Sauts moyens",          bsl.get('mean_hops'),         flt.get('mean_hops')),
        ("Stockage moy. (s)",     bsl.get('mean_store_s'),      flt.get('mean_store_s')),
        ("Tx moy. (s)",           bsl.get('mean_tx_s'),         flt.get('mean_tx_s')),
    ]

    for label, b_val, f_val in rows:
        print(
            f"  {label:<30} {_fmt(b_val):>10} {_fmt(f_val):>10}"
            f" {_delta(b_val, f_val):>10}"
        )

    print("=" * W)


def comparison_dataframe(comparison: dict):
    """Return a pandas DataFrame of the comparison (for notebook display)."""
    import pandas as pd

    bsl = comparison['baseline']
    flt = comparison['failure']

    def _delta_str(b, f):
        if b is None or f is None:
            return "N/A"
        if b == 0:
            return "+inf" if f > 0 else "=0"
        return f"{(f - b) / b * 100:+.1f}%"

    rows = [
        ("Taux de livraison (%)", bsl['delivery_ratio'] * 100, flt['delivery_ratio'] * 100),
        ("PDUs livres",           bsl['n_delivered'],           flt['n_delivered']),
        ("Goodput (bps)",         bsl['goodput_bps'],           flt['goodput_bps']),
        ("Latence moy. (s)",      bsl.get('mean_latency_s'),    flt.get('mean_latency_s')),
        ("Latence min (s)",       bsl.get('min_latency_s'),     flt.get('min_latency_s')),
        ("Latence max (s)",       bsl.get('max_latency_s'),     flt.get('max_latency_s')),
        ("Ecart-type lat. (s)",   bsl.get('std_latency_s'),     flt.get('std_latency_s')),
        ("Sauts moyens",          bsl.get('mean_hops'),         flt.get('mean_hops')),
        ("Stockage moy. (s)",     bsl.get('mean_store_s'),      flt.get('mean_store_s')),
        ("Tx moy. (s)",           bsl.get('mean_tx_s'),         flt.get('mean_tx_s')),
    ]

    def _fmt(v):
        return round(v, 4) if v is not None else None

    df = pd.DataFrame(
        {
            "Avant":     [_fmt(b) for _, b, _ in rows],
            "Apres":     [_fmt(f) for _, _, f in rows],
            "Delta (%)": [_delta_str(b, f) for _, b, f in rows],
        },
        index=[label for label, _, _ in rows],
    )
    df.index.name = "Metrique"
    return df
