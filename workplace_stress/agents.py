"""Employee agent: an investment-bank worker whose stress evolves weekly."""

from __future__ import annotations

from enum import Enum

import mesa
import numpy as np


class StressState(str, Enum):
    HEALTHY = "Healthy"
    STRESSED = "Stressed"
    BURNOUT = "Burnout"


# Seniority ladder: base weekly workload (relative), revenue contribution
# (in $k per week at peak performance), and operational-risk exposure
# (how large a mistake by this person can be, in $k).
ROLES = {
    "Analyst":   {"workload": 1.25, "value": 18.0,  "exposure": 40.0,  "share": 0.38},
    "Associate": {"workload": 1.15, "value": 30.0,  "exposure": 80.0,  "share": 0.27},
    "VP":        {"workload": 1.00, "value": 55.0,  "exposure": 150.0, "share": 0.18},
    "Director":  {"workload": 0.90, "value": 90.0,  "exposure": 250.0, "share": 0.10},
    "MD":        {"workload": 0.85, "value": 160.0, "exposure": 400.0, "share": 0.07},
}

# Desks differ in how sensitive their workload is to market turbulence and
# how large their risk exposure is.
DESKS = {
    "M&A":       {"market_beta": 0.6, "risk_mult": 0.6},
    "Trading":   {"market_beta": 1.6, "risk_mult": 2.0},
    "DCM/ECM":   {"market_beta": 1.0, "risk_mult": 0.9},
    "Research":  {"market_beta": 0.8, "risk_mult": 0.4},
    "Risk/Ops":  {"market_beta": 1.2, "risk_mult": 1.2},
}

# Thresholds on the continuous stress level that define discrete states.
STRESSED_AT = 0.40
BURNOUT_AT = 0.75
# Stress level at which performance peaks (Yerkes–Dodson inverted U).
OPTIMAL_STRESS = 0.30


class Employee(mesa.Agent):
    """A single bank employee.

    Stress ``s`` lives in [0, 1]. Each week it is pushed up by long hours and
    by stressed colleagues (social contagion), and pulled down by recovery,
    which depends on personal resilience and firm wellbeing support.
    """

    def __init__(self, model, bank, role: str, desk: str, tenure_weeks: int = 0):
        super().__init__(model)
        self.bank = bank
        self.role = role
        self.desk = desk
        self.tenure_weeks = tenure_weeks

        rng = model.rng
        self.resilience = float(np.clip(rng.normal(0.5, 0.15), 0.1, 0.9))
        self.stress = float(np.clip(rng.beta(2, 5), 0.0, 1.0))
        self.burnout_weeks = 0

        self.hours = 0.0
        self.output = 0.0
        self.op_loss = 0.0
        self.state = self._classify()

    # ------------------------------------------------------------------ helpers
    def _classify(self) -> StressState:
        if self.stress >= BURNOUT_AT:
            return StressState.BURNOUT
        if self.stress >= STRESSED_AT:
            return StressState.STRESSED
        return StressState.HEALTHY

    def performance(self) -> float:
        """Inverted-U productivity: some pressure helps, too much destroys."""
        perf = np.exp(-(((self.stress - OPTIMAL_STRESS) / 0.35) ** 2))
        # Newer hires ramp up over ~6 months.
        ramp = min(1.0, 0.5 + self.tenure_weeks / 52)
        return float(perf * ramp)

    def neighbor_stress(self) -> float:
        neighbors = [
            self.model.employees_by_id[n]
            for n in self.model.network.neighbors(self.unique_id)
            if n in self.model.employees_by_id
        ]
        if not neighbors:
            return self.stress
        return float(np.mean([n.stress for n in neighbors]))

    # --------------------------------------------------------------------- step
    def step(self):
        m = self.model
        rng = m.rng
        role = ROLES[self.role]
        desk = DESKS[self.desk]

        # 1. Workload: role baseline x desk vacancies x market turbulence.
        market_push = 1.0 + desk["market_beta"] * (m.market_vol / m.base_vol - 1.0) * 0.25
        workload = role["workload"] * self.bank.desk_load[self.desk] * market_push
        self.hours = float(np.clip(55 * workload + rng.normal(0, 4), 40, 110))
        if m.hours_cap is not None:
            self.hours = min(self.hours, m.hours_cap)

        # 2. Stress dynamics.
        overwork = max(0.0, self.hours - 60) / 40          # 0 at 60h, 1 at 100h
        # Negative affect spreads more readily than calm: only colleagues who
        # are more stressed than you pull you up.
        contagion = m.contagion * max(0.0, self.neighbor_stress() - self.stress)
        recovery = m.recovery_rate * (0.5 + self.resilience) * (1 + m.wellbeing_support)
        recovery *= max(0.0, 1 - overwork)                  # no recovery when buried
        shock = rng.normal(0, 0.03)
        self.stress += 0.10 * overwork + contagion - recovery * self.stress + shock
        self.stress = float(np.clip(self.stress, 0.0, 1.0))

        self.state = self._classify()
        self.burnout_weeks = self.burnout_weeks + 1 if self.state is StressState.BURNOUT else 0
        self.tenure_weeks += 1

        # 3. Output: burnt-out staff are sometimes absent entirely.
        absent = self.state is StressState.BURNOUT and rng.random() < 0.25
        self.output = 0.0 if absent else role["value"] * self.performance()

        # 4. Operational risk: errors grow convexly with stress.
        p_error = m.base_error_rate * (1 + 10 * self.stress ** 2)
        self.op_loss = 0.0
        if not absent and rng.random() < p_error:
            self.op_loss = float(role["exposure"] * desk["risk_mult"] * rng.lognormal(0, 1.0))

        # 5. Attrition: quit hazard rises with sustained burnout.
        p_quit = m.base_quit_rate + 0.02 * self.burnout_weeks
        if rng.random() < p_quit:
            m.schedule_exit(self)
