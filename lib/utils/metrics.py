from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

from ..classes.Satellite import Satellite
from ..classes.NetworkSimulation import NetworkSimulation, PDU, EpidemicRouter, LinkStateRouter

SPEED_OF_LIGHT = 3e8  # m/s

# Calibré depuis traces.csv :
#   - Période orbitale ≈ 1792 pas (analyse des changements de signe sur X)
#   - LEO à 500 km d'altitude → T_orbit ≈ 94.6 min
#   - step_duration = 94.6 * 60 / 1792 ≈ 3.2 s
DEFAULT_STEP_DURATION_S = 1.0


@dataclass
class MetricsConfig:
    step_duration_s: float = DEFAULT_STEP_DURATION_S
    link_rate_bps:   float = 9_600   # UHF nanosatellite (typique CubeSat)
    pdu_size_bytes:  float = 1_024   # 1 KB par bundle

    @property
    def tx_delay_per_hop_s(self) -> float:
        """Temps pour transmettre un PDU complet sur un lien."""
        return self.pdu_size_bytes * 8 / self.link_rate_bps

    @property
    def max_valid_pdu_bytes(self) -> float:
        """Taille maximale de PDU transmissible en un pas de simulation."""
        return self.link_rate_bps * self.step_duration_s / 8

    @property
    def is_physically_valid(self) -> bool:
        """True si le PDU peut être transmis en un seul pas (modèle valide)."""
        return self.tx_delay_per_hop_s <= self.step_duration_s


@dataclass
class PDUResult:
    pdu_id:         int
    delivered:      bool
    injection_t:    int
    delivery_t:     int | None   = None
    trace:          list[str]    = field(default_factory=list)

    # décomposition de la latence
    latency_steps:  int   | None = None
    latency_s:      float | None = None   # latence E2E corrigée
    store_delay_s:  float | None = None   # temps d'attente (stockage)
    tx_delay_s:     float | None = None   # temps de transmission cumulé
    prop_delay_s:   float | None = None   # délai de propagation (lumière)
    hops:           int   | None = None


def _prop_delay(trace: list[str], sat_map: dict[str, Satellite], t: int) -> float:
    """Délai de propagation lumière le long de la trace à l'instant t."""
    delay = 0.0
    max_t = len(next(iter(sat_map.values())).list_coordinates) - 1
    ti = min(t, max_t)
    for a, b in zip(trace, trace[1:]):
        if a in sat_map and b in sat_map:
            p1 = sat_map[a].list_coordinates[ti].point
            p2 = sat_map[b].list_coordinates[ti].point
            delay += p1.distance(p2) / SPEED_OF_LIGHT
    return delay


def _corrected_latency(latency_steps: int, hops: int, config: MetricsConfig,
                        prop_delay_s: float) -> tuple[float, float, float, float]:
    """
    Calcule la latence E2E corrigée et sa décomposition.

    Le modèle de simulation suppose que chaque hop prend exactement 1 pas.
    C'est valide si tx_per_hop <= step_duration. Pour de gros PDUs, on corrige :

        L_E2E = (latency_steps - hops) * step_s   <- attente pour contacts
              + hops * max(tx_per_hop_s, step_s)  <- transmission réelle par hop
              + prop_s

    Pour de petits PDUs (tx <= step) : L_E2E = latency_steps * step_s (inchangé).
    Pour de gros PDUs (tx > step)   : L_E2E > latency_steps * step_s (corrigé).
    """
    step_s      = config.step_duration_s
    tx_hop_s    = config.tx_delay_per_hop_s
    wait_steps  = latency_steps - hops          # pas passés à attendre un contact

    store_delay_s = wait_steps * step_s
    tx_delay_s    = tx_hop_s * hops
    latency_s     = store_delay_s + tx_delay_s + prop_delay_s
    return latency_s, store_delay_s, tx_delay_s, prop_delay_s


def _run_metrics(
    satellites:          list[Satellite],
    source_name:         str,
    dest_name:           str,
    router_class,
    n_pdus:              int               = 5,
    injection_interval:  int               = 30,
    config:              MetricsConfig | None = None,
    max_steps:           int               = 500,
) -> dict:
    """Cœur générique : injecte n_pdus PDUs et mesure latence + goodput E2E.

    Modèle de latence : L = L_store + L_tx + L_prop
      - L_store : attente d'un contact (dominant en DTN)
      - L_tx    : transmission physique cumulée sur tous les sauts
      - L_prop  : propagation lumière (négligeable à < 30 km)

    Goodput = bits livrés / durée totale, plafonné au débit du lien.
    """
    config  = config or MetricsConfig()
    sat_map = {s.name: s for s in satellites}

    scenario = [
        (i * injection_interval, PDU(i, source_name, dest_name))
        for i in range(n_pdus)
    ]

    sim      = NetworkSimulation(satellites, router_class, scenario)
    delivered: dict = {}
    seen:      set  = set()

    for t in range(max_steps):
        for pdu in sim.routers[dest_name].pdus_where_this_router_is_the_destination:
            if pdu.id not in seen:
                delivered[pdu.id] = {'delivery_t': t, 'trace': list(pdu.trace)}
                seen.add(pdu.id)
        if len(seen) == n_pdus:
            break
        sim.next()

    # ── Métriques par PDU ──────────────────────────────────────────────────────
    results: list[PDUResult] = []
    for i in range(n_pdus):
        inj_t = i * injection_interval
        if i not in delivered:
            results.append(PDUResult(pdu_id=i, delivered=False, injection_t=inj_t))
            continue

        del_t  = delivered[i]['delivery_t']
        trace  = delivered[i]['trace']
        hops   = len(trace) - 1

        latency_steps = del_t - inj_t
        prop_s        = _prop_delay(trace, sat_map, del_t)
        lat_s, store_s, tx_s, prop_s = _corrected_latency(
            latency_steps, hops, config, prop_s
        )

        results.append(PDUResult(
            pdu_id        = i,
            delivered     = True,
            injection_t   = inj_t,
            delivery_t    = del_t,
            trace         = trace,
            latency_steps = latency_steps,
            latency_s     = lat_s,
            hops          = hops,
            tx_delay_s    = tx_s,
            prop_delay_s  = prop_s,
            store_delay_s = store_s,
        ))

    # ── Métriques agrégées ─────────────────────────────────────────────────────
    ok = [r for r in results if r.delivered]
    delivery_ratio = len(ok) / n_pdus

    total_time_s = max_steps * config.step_duration_s
    raw_goodput  = len(ok) * config.pdu_size_bytes * 8 / total_time_s
    goodput_bps  = min(raw_goodput, config.link_rate_bps)

    agg = dict(
        delivery_ratio = delivery_ratio,
        goodput_bps    = goodput_bps,
        total_time_s   = total_time_s,
        n_delivered    = len(ok),
        n_pdus         = n_pdus,
    )
    if ok:
        agg.update(
            mean_latency_s = float(np.mean([r.latency_s     for r in ok])),
            std_latency_s  = float(np.std ([r.latency_s     for r in ok])),
            min_latency_s  = float(np.min ([r.latency_s     for r in ok])),
            max_latency_s  = float(np.max ([r.latency_s     for r in ok])),
            mean_hops      = float(np.mean([r.hops          for r in ok])),
            mean_store_s   = float(np.mean([r.store_delay_s for r in ok])),
            mean_tx_s      = float(np.mean([r.tx_delay_s    for r in ok])),
            mean_prop_s    = float(np.mean([r.prop_delay_s  for r in ok])),
        )

    return {'config': config, 'results': results, **agg}


def run_epidemic_metrics(
    satellites:          list[Satellite],
    source_name:         str,
    dest_name:           str,
    n_pdus:              int               = 5,
    injection_interval:  int               = 30,
    config:              MetricsConfig | None = None,
    max_steps:           int               = 500,
) -> dict:
    """Epidemic routing — store-carry-forward multi-copie."""
    return _run_metrics(
        satellites, source_name, dest_name, EpidemicRouter,
        n_pdus=n_pdus, injection_interval=injection_interval,
        config=config, max_steps=max_steps,
    )


def run_linkstate_metrics(
    satellites:          list[Satellite],
    source_name:         str,
    dest_name:           str,
    n_pdus:              int               = 5,
    injection_interval:  int               = 30,
    config:              MetricsConfig | None = None,
    max_steps:           int               = 500,
) -> dict:
    """Link-state routing — copie unique, plus court chemin BFS."""
    return _run_metrics(
        satellites, source_name, dest_name, LinkStateRouter,
        n_pdus=n_pdus, injection_interval=injection_interval,
        config=config, max_steps=max_steps,
    )


def run_algo_comparison(
    satellites:          list[Satellite],
    source_name:         str,
    dest_name:           str,
    n_pdus:              int               = 8,
    injection_interval:  int               = 20,
    config:              MetricsConfig | None = None,
    max_steps:           int               = 500,
    failure_config=None,
) -> dict:
    """Compare Epidemic et Link-State, avec ou sans pannes.

    Retourne un dict contenant :
      'epidemic_normal', 'linkstate_normal'
      'epidemic_failure', 'linkstate_failure'  (si failure_config fourni)
      'failed_satellites', 'failure_config'
    """
    common = dict(
        n_pdus=n_pdus, injection_interval=injection_interval,
        config=config, max_steps=max_steps,
    )

    ep_normal = _run_metrics(satellites, source_name, dest_name, EpidemicRouter,   **common)
    ls_normal = _run_metrics(satellites, source_name, dest_name, LinkStateRouter,  **common)

    result: dict = {
        'epidemic_normal':  ep_normal,
        'linkstate_normal': ls_normal,
        'failed_satellites': [],
        'failure_config': failure_config,
    }

    if failure_config is not None:
        from .failure import apply_failures, _resolve_failed_names
        failed_names = _resolve_failed_names(satellites, failure_config)
        t_end = (
            failure_config.t_end
            if failure_config.t_end is not None
            else len(satellites[0].list_coordinates) - 1
        )
        broken = apply_failures(satellites, failed_names, failure_config.t_start, t_end)

        result['epidemic_failure']  = _run_metrics(broken, source_name, dest_name, EpidemicRouter,  **common)
        result['linkstate_failure'] = _run_metrics(broken, source_name, dest_name, LinkStateRouter, **common)
        result['failed_satellites'] = failed_names

    return result


def print_algo_comparison_table(
    comparison: dict,
    source: str = "",
    dest:   str = "",
) -> None:
    """Tableau comparatif Epidemic vs Link-State, avec colonne panne si disponible."""
    ep  = comparison['epidemic_normal']
    ls  = comparison['linkstate_normal']
    has_failure = 'epidemic_failure' in comparison
    failed = comparison.get('failed_satellites', [])

    W = 80

    def _f(v):
        return "N/A" if v is None else f"{v:.2f}"

    def _d(ref, new):
        if ref is None or new is None:
            return "N/A"
        if ref == 0:
            return "+inf" if new > 0 else "=0"
        return f"{(new - ref) / ref * 100:+.1f}%"

    print("=" * W)
    print("  COMPARAISON ALGORITHMES : Epidemic vs Link-State Routing")
    print("=" * W)
    if source and dest:
        print(f"  Source --> Dest : {source} --> {dest}")

    if has_failure:
        ep_f = comparison['epidemic_failure']
        ls_f = comparison['linkstate_failure']
        sat_list = ', '.join(failed) if failed else "aucun"
        print(f"  Pannes          : {sat_list}")
        print("-" * W)
        print(
            f"  {'Metrique':<26}"
            f" {'Ep.(norm)':>10} {'LS (norm)':>10}"
            f" {'Ep.(panne)':>10} {'LS(panne)':>10}"
        )
        print("-" * W)

        rows = [
            ("Taux livraison (%)", ep['delivery_ratio']*100, ls['delivery_ratio']*100,
                                   ep_f['delivery_ratio']*100, ls_f['delivery_ratio']*100),
            ("PDUs livres",        ep['n_delivered'],         ls['n_delivered'],
                                   ep_f['n_delivered'],        ls_f['n_delivered']),
            ("Goodput (bps)",      ep['goodput_bps'],         ls['goodput_bps'],
                                   ep_f['goodput_bps'],        ls_f['goodput_bps']),
            ("Latence moy. (s)",   ep.get('mean_latency_s'),  ls.get('mean_latency_s'),
                                   ep_f.get('mean_latency_s'), ls_f.get('mean_latency_s')),
            ("Latence min (s)",    ep.get('min_latency_s'),   ls.get('min_latency_s'),
                                   ep_f.get('min_latency_s'),  ls_f.get('min_latency_s')),
            ("Latence max (s)",    ep.get('max_latency_s'),   ls.get('max_latency_s'),
                                   ep_f.get('max_latency_s'),  ls_f.get('max_latency_s')),
            ("Sauts moyens",       ep.get('mean_hops'),       ls.get('mean_hops'),
                                   ep_f.get('mean_hops'),      ls_f.get('mean_hops')),
            ("Stockage moy. (s)",  ep.get('mean_store_s'),    ls.get('mean_store_s'),
                                   ep_f.get('mean_store_s'),   ls_f.get('mean_store_s')),
        ]
        for label, *vals in rows:
            print(f"  {label:<26}" + "".join(f" {_f(v):>10}" for v in vals))

    else:
        print("-" * W)
        print(f"  {'Metrique':<30} {'Epidemic':>12} {'Link-State':>12} {'Delta(LS/E)':>12}")
        print("-" * W)

        rows = [
            ("Taux de livraison (%)", ep['delivery_ratio']*100, ls['delivery_ratio']*100),
            ("PDUs livres",           ep['n_delivered'],         ls['n_delivered']),
            ("Goodput (bps)",         ep['goodput_bps'],         ls['goodput_bps']),
            ("Latence moy. (s)",      ep.get('mean_latency_s'),  ls.get('mean_latency_s')),
            ("Latence min (s)",       ep.get('min_latency_s'),   ls.get('min_latency_s')),
            ("Latence max (s)",       ep.get('max_latency_s'),   ls.get('max_latency_s')),
            ("Sauts moyens",          ep.get('mean_hops'),       ls.get('mean_hops')),
            ("Stockage moy. (s)",     ep.get('mean_store_s'),    ls.get('mean_store_s')),
        ]
        for label, ep_val, ls_val in rows:
            print(f"  {label:<30} {_f(ep_val):>12} {_f(ls_val):>12} {_d(ep_val, ls_val):>12}")

    print("=" * W)


def algo_comparison_dataframe(comparison: dict):
    """Retourne un DataFrame pandas (affichage HTML en notebook)."""
    import pandas as pd

    ep  = comparison['epidemic_normal']
    ls  = comparison['linkstate_normal']
    has_failure = 'epidemic_failure' in comparison

    def _d(ref, new):
        if ref is None or new is None:
            return "N/A"
        if ref == 0:
            return "+inf" if new > 0 else "=0"
        return f"{(new - ref) / ref * 100:+.1f}%"

    rows = [
        ("Taux de livraison (%)", ep['delivery_ratio']*100, ls['delivery_ratio']*100),
        ("PDUs livres",           ep['n_delivered'],         ls['n_delivered']),
        ("Goodput (bps)",         ep['goodput_bps'],         ls['goodput_bps']),
        ("Latence moy. (s)",      ep.get('mean_latency_s'),  ls.get('mean_latency_s')),
        ("Latence min (s)",       ep.get('min_latency_s'),   ls.get('min_latency_s')),
        ("Latence max (s)",       ep.get('max_latency_s'),   ls.get('max_latency_s')),
        ("Sauts moyens",          ep.get('mean_hops'),       ls.get('mean_hops')),
        ("Stockage moy. (s)",     ep.get('mean_store_s'),    ls.get('mean_store_s')),
    ]

    data: dict = {
        "Epidemic":       [r[1] for r in rows],
        "Link-State":     [r[2] for r in rows],
        "Delta (LS/Ep)":  [_d(r[1], r[2]) for r in rows],
    }

    if has_failure:
        ep_f = comparison['epidemic_failure']
        ls_f = comparison['linkstate_failure']
        rows_f = [
            ep_f['delivery_ratio']*100, ep_f['n_delivered'],      ep_f['goodput_bps'],
            ep_f.get('mean_latency_s'), ep_f.get('min_latency_s'), ep_f.get('max_latency_s'),
            ep_f.get('mean_hops'),      ep_f.get('mean_store_s'),
        ]
        rows_ls_f = [
            ls_f['delivery_ratio']*100, ls_f['n_delivered'],      ls_f['goodput_bps'],
            ls_f.get('mean_latency_s'), ls_f.get('min_latency_s'), ls_f.get('max_latency_s'),
            ls_f.get('mean_hops'),      ls_f.get('mean_store_s'),
        ]
        data["Ep. (panne)"]         = rows_f
        data["LS  (panne)"]         = rows_ls_f
        data["Delta (LS/Ep panne)"] = [_d(ep_v, ls_v) for ep_v, ls_v in zip(rows_f, rows_ls_f)]

    df = pd.DataFrame(data, index=[r[0] for r in rows])
    df.index.name = "Metrique"
    return df


def algo_comparison_dataframe_full(
    comp_normal: dict,
    comp_manual: dict | None = None,
    comp_random: dict | None = None,
):
    """Tableau unifié : normal + pannes manuelles + pannes aléatoires (Epidemic & LSP).

    Args:
        comp_normal: résultat de run_algo_comparison sans failure_config.
        comp_manual: résultat de run_algo_comparison avec pannes volontaires.
        comp_random: résultat de run_algo_comparison avec pannes aléatoires.
    """
    import pandas as pd

    METRIC_LABELS = [
        "Taux de livraison (%)",
        "PDUs livres",
        "Goodput (bps)",
        "Latence moy. (s)",
        "Latence min (s)",
        "Latence max (s)",
        "Sauts moyens",
        "Stockage moy. (s)",
    ]

    def _d(ref, new):
        if ref is None or new is None:
            return "N/A"
        if ref == 0:
            return "+inf" if new > 0 else "=0"
        return f"{(new - ref) / ref * 100:+.1f}%"

    def _extract(m: dict) -> list:
        return [
            m['delivery_ratio'] * 100,
            m['n_delivered'],
            m['goodput_bps'],
            m.get('mean_latency_s'),
            m.get('min_latency_s'),
            m.get('max_latency_s'),
            m.get('mean_hops'),
            m.get('mean_store_s'),
        ]

    ep_n = _extract(comp_normal['epidemic_normal'])
    ls_n = _extract(comp_normal['linkstate_normal'])

    data: dict = {
        "Epidemic":      ep_n,
        "Link-State":    ls_n,
        "Delta (LS/Ep)": [_d(e, l) for e, l in zip(ep_n, ls_n)],
    }

    if comp_manual is not None and 'epidemic_failure' in comp_manual:
        ep_m = _extract(comp_manual['epidemic_failure'])
        ls_m = _extract(comp_manual['linkstate_failure'])
        data["Ep. (manu.)"]   = ep_m
        data["LS  (manu.)"]   = ls_m
        data["Delta (manu.)"] = [_d(e, l) for e, l in zip(ep_m, ls_m)]

    if comp_random is not None and 'epidemic_failure' in comp_random:
        ep_r = _extract(comp_random['epidemic_failure'])
        ls_r = _extract(comp_random['linkstate_failure'])
        data["Ep. (alea.)"]   = ep_r
        data["LS  (alea.)"]   = ls_r
        data["Delta (alea.)"] = [_d(e, l) for e, l in zip(ep_r, ls_r)]

    df = pd.DataFrame(data, index=METRIC_LABELS)
    df.index.name = "Metrique"
    return df


def print_metrics_report(m: dict) -> None:
    cfg = m['config']

    print("=" * 62)
    print("  METRIQUES EPIDEMIC ROUTING -- bout en bout")
    print("=" * 62)
    print(f"  Duree d'un pas     : {cfg.step_duration_s} s")
    print(f"  Debit du lien      : {cfg.link_rate_bps/1000:.1f} kbps")
    print(f"  Taille PDU         : {cfg.pdu_size_bytes:.0f} B")
    print(f"  Tx par hop         : {cfg.tx_delay_per_hop_s:.3f} s")
    print(f"  PDU max valide     : {cfg.max_valid_pdu_bytes:.0f} B  "
          f"({'OK' if cfg.is_physically_valid else 'DEPASSE'})")

    if not cfg.is_physically_valid:
        factor = cfg.tx_delay_per_hop_s / cfg.step_duration_s
        print()
        print(f"  [!] PDU trop grand pour le modele discret (x{factor:.1f} le pas).")
        print(f"      La latence est corrigee : L = attente + hops x tx_reel.")
        print(f"      Le goodput est plafonne au debit du lien ({cfg.link_rate_bps:.0f} bps).")

    print()
    print(f"  Duree simulation   : {m['total_time_s']/60:.1f} min")
    print(f"  PDUs envoyes       : {m['n_pdus']}")
    print(f"  PDUs recus         : {m['n_delivered']}")
    print(f"  Taux de livraison  : {m['delivery_ratio']*100:.1f} %")
    print(f"  Goodput            : {m['goodput_bps']:.2f} bps  "
          f"({m['goodput_bps']/cfg.link_rate_bps*100:.2f} % du debit lien)")
    print()
    if 'mean_latency_s' in m:
        ml = m['mean_latency_s']
        def pct(v): return f"({v/ml*100:.1f} %)" if ml > 0 else ""
        unit = "s" if ml < 3600 else "min"
        scale = 1 if unit == "s" else 60

        print(f"  Latence moyenne    : {ml/scale:.1f} {unit}  +/- {m['std_latency_s']/scale:.1f} {unit}")
        print(f"  Latence min/max    : {m['min_latency_s']/scale:.1f} / {m['max_latency_s']/scale:.1f} {unit}")
        print(f"    dont stockage    : {m['mean_store_s']/scale:.1f} {unit}  {pct(m['mean_store_s'])}")
        print(f"    dont tx          : {m['mean_tx_s']/scale:.1f} {unit}  {pct(m['mean_tx_s'])}")
        print(f"    dont prop.       : {m['mean_prop_s']*1000:.3f} ms  {pct(m['mean_prop_s'])}")
        print(f"  Sauts moyens       : {m['mean_hops']:.1f}")
    print()
    print(f"  {'ID':>3}  {'Inj.':>5}  {'Livr.':>6}  {'Latence':>12}  {'Hops':>4}  Trace")
    print("  " + "-" * 58)
    for r in m['results']:
        if r.delivered:
            lat = r.latency_s
            u   = "s" if lat < 3600 else "min"
            lv  = lat if u == "s" else lat / 60
            path = " -> ".join(r.trace)
            print(f"  {r.pdu_id:>3}  {r.injection_t:>5}  {r.delivery_t:>6}  "
                  f"{lv:>9.1f} {u}  {r.hops:>4}  {path}")
        else:
            print(f"  {r.pdu_id:>3}  {r.injection_t:>5}  {'--':>6}  "
                  f"{'non livre':>12}")
    print("=" * 62)
