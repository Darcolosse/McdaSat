from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

from ..classes.Satellite import Satellite
from ..classes.NetworkSimulation import NetworkSimulation, PDU, EpidemicRouter

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


def run_epidemic_metrics(
    satellites:          list[Satellite],
    source_name:         str,
    dest_name:           str,
    n_pdus:              int               = 5,
    injection_interval:  int               = 30,
    config:              MetricsConfig | None = None,
    max_steps:           int               = 500,
) -> dict:
    """
    Injecte n_pdus PDUs à intervalles réguliers et mesure latence + goodput E2E.

    Modèle de latence : L = L_store + L_tx + L_prop
      - L_store : temps passé à attendre un contact (dominant en DTN)
      - L_tx    : temps de transmission physique cumulé sur tous les sauts
      - L_prop  : délai de propagation lumière (négligeable à < 30 km)

    Goodput = bits livrés / durée totale, plafonné au débit du lien.
    """
    config  = config or MetricsConfig()
    sat_map = {s.name: s for s in satellites}

    scenario = [
        (i * injection_interval, PDU(i, source_name, dest_name))
        for i in range(n_pdus)
    ]

    sim      = NetworkSimulation(satellites, EpidemicRouter, scenario)
    delivered = {}
    seen      = set()

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

    # Goodput = bits livrés / durée totale, plafonné au débit physique du lien
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
            mean_latency_s = float(np.mean([r.latency_s    for r in ok])),
            std_latency_s  = float(np.std ([r.latency_s    for r in ok])),
            min_latency_s  = float(np.min ([r.latency_s    for r in ok])),
            max_latency_s  = float(np.max ([r.latency_s    for r in ok])),
            mean_hops      = float(np.mean([r.hops         for r in ok])),
            mean_store_s   = float(np.mean([r.store_delay_s for r in ok])),
            mean_tx_s      = float(np.mean([r.tx_delay_s   for r in ok])),
            mean_prop_s    = float(np.mean([r.prop_delay_s for r in ok])),
        )

    return {'config': config, 'results': results, **agg}


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
