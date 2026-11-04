"""
Disruption layer for the stage-model simulator (Phase 5).

Generic mechanisms, mapped onto a log by `<log>_disruptions.json`:
  outage         a service is down for an uncertain time. While it is down, the
                 blocked work items cannot start and the blocked stage transitions
                 cannot happen; the case fails and the agent decides.
  routing_surge  for a while, a stage exit (e.g. validation -> incomplete) is more
                 likely; every case routed there during the surge is a decision.
  arrival_surge  demand spike: for a while, applications arrive at `arrival_factor`
                 times the normal (hour-of-week) rate; nothing is blocked, the extra
                 work queues up.
Plus the SLA-risk alert (D6): a blocked case left waiting gets a second decision
when its slack (stage deadline - case age) falls to a threshold.

One scenario per episode (`scenario`, `level`), fired at `start(onset)`. The
length is drawn here from the level's distribution, or fixed by `duration_h`; either
way it is never exposed to the agent.
"""
import json

import numpy as np


def sample_hours(spec, rng):
    """Duration in hours from {"uniform": [a, b]} or {"mixture": [{"p":, "uniform":}, ...]}."""
    if 'mixture' in spec:
        parts = spec['mixture']
        k = rng.choice(len(parts), p=np.array([p['p'] for p in parts]) / sum(p['p'] for p in parts))
        return sample_hours(parts[k], rng)
    a, b = spec['uniform']
    return float(rng.uniform(a, b))


def scenario_key(scenario, level, duration_h=None):
    """'D1_offer_outage:2' (length drawn) or 'D1_offer_outage:2:24h' (fixed length)."""
    key = f'{scenario}:{level}'
    return key if duration_h is None else f'{key}:{duration_h:g}h'


def cycle_key(scenario, level):
    """Key of a scenario's normalisation pooled over its training-cycle cells."""
    return f'{scenario}:{level}:cycle'


class DisruptionLayer:
    def __init__(self, config, scenario, level, deadlines_s, rng, duration_h=None):
        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        self.cfg = config
        self.name, self.level = scenario, str(level)
        self.spec = config['scenarios'][scenario]
        self.params = self.spec['levels'][self.level]
        self.fixed_duration_h = duration_h
        self.mechanism = self.spec['mechanism']
        self.failure_type = self.spec['failure_type']
        self.deadlines = deadlines_s
        self.rng = rng
        alert = config.get('sla_alert', {})
        self.alert_type = alert.get('failure_type', 'sla_risk')
        self.alert_threshold_s = alert['slack_threshold_h'] * 3600 if alert.get('enabled') else None
        blocks = self.spec.get('blocks', {})
        self.blocked_items = set(blocks.get('work_items', []))
        self.blocked_transitions = {tuple(t) for t in blocks.get('transitions', [])}
        self.active = False
        self.onset = self.end = None
        self.duration_s = None
        self.restored = None     # simpy event, triggered when the disruption ends

    @property
    def key(self):
        return scenario_key(self.name, self.level, self.fixed_duration_h)

    def start(self, env, onset):
        """Schedule the disruption at `onset` (sim seconds); returns its duration."""
        self.env = env
        hours = self.fixed_duration_h
        if hours is None:
            hours = sample_hours(self.params['duration_h'], self.rng)
        self.duration_s = float(hours) * 3600
        self.onset, self.end = onset, onset + self.duration_s
        self.restored = env.event()
        env.process(self._run())
        return self.duration_s

    def _run(self):
        yield self.env.timeout(max(0.0, self.onset - self.env.now))
        self.active = True
        yield self.env.timeout(self.duration_s)
        self.active = False
        self.restored.succeed()

    # ---------------------------------------------------------------- hooks
    def arrival_factor(self):
        """Multiplier of the arrival rate (demand spike), 1 otherwise."""
        if self.active and self.mechanism == 'arrival_surge':
            return float(self.params['arrival_factor'])
        return 1.0

    def blocks_session(self, item):
        if self.active and self.mechanism == 'outage' and item in self.blocked_items:
            return self.failure_type
        return None

    def blocks_transition(self, stage, nxt):
        if self.active and self.mechanism == 'outage' and (stage, nxt) in self.blocked_transitions:
            return self.failure_type
        return None

    def modify_exits(self, stage, exits):
        """Exit probabilities during a routing surge (target exit x (1 + magnitude),
        capped at 0.95, the others scaled down to keep the sum)."""
        if not (self.active and self.mechanism == 'routing_surge' and stage == self.spec['stage']):
            return exits
        tgt = self.spec['exit']
        p_old = exits.get(tgt, 0.0)
        tot = sum(exits.values())
        p_new = min(0.95 * tot, p_old * (1 + self.params['magnitude']))
        rest = tot - p_old
        scale = (tot - p_new) / rest if rest > 0 else 0.0
        return {k: (p_new if k == tgt else v * scale) for k, v in exits.items()}

    def surge_decision(self, stage, nxt):
        """Failure type if a case routed stage -> nxt during the surge needs a decision."""
        if (self.active and self.mechanism == 'routing_surge' and stage == self.spec['stage']
                and nxt == self.spec['exit']):
            return self.failure_type
        return None

    def alert_delay(self, arrival, stage, now):
        """Seconds until the SLA-risk alert for a case of this age in this stage
        (None if alerts are off or the stage has no deadline)."""
        if self.alert_threshold_s is None or stage not in self.deadlines:
            return None
        return max(0.0, arrival + self.deadlines[stage] - self.alert_threshold_s - now)
