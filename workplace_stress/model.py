"""WorkplaceStressModel: banks of stressed employees inside a shared market.

The feedback loop the model studies:

    market turbulence -> heavier workload -> employee stress
      -> stress contagion over the office network
      -> lower productivity, more operational errors, more quits
      -> vacancies shift work onto remaining colleagues (cascade)
      -> noisier bank P&L (firm volatility)
      -> bank losses feed back into market volatility (macro volatility)

One step is one week.
"""

from __future__ import annotations

from collections import defaultdict

import mesa
import networkx as nx
import numpy as np

from .agents import DESKS, ROLES, Employee, StressState


class Bank:
    """A firm: a roster of employees plus its P&L and capital."""

    def __init__(self, bank_id: int, headcount: int, capital: float):
        self.bank_id = bank_id
        self.capital = capital               # $k
        self.employees: list[Employee] = []
        self.target: dict[tuple[str, str], int] = defaultdict(int)  # (desk, role) -> seats
        self.desk_target: dict[str, int] = defaultdict(int)
        self.desk_load: dict[str, float] = {d: 1.0 for d in DESKS}
        self.ideal_revenue = 0.0             # weekly revenue at full health
        self.pnl = 0.0
        self.ret = 0.0
        self.returns: list[float] = []
        self.op_losses = 0.0
        self.headcount = headcount
        self.cost_ratio = 0.78
        self.vacancy_passthrough = 0.6
        self.idio_returns: list[float] = []

    def update_desk_load(self):
        counts = defaultdict(int)
        for e in self.employees:
            counts[e.desk] += 1
        for desk in DESKS:
            target = self.desk_target[desk]
            have = max(counts[desk], 1)
            # Vacancies push work onto fewer people; some of it is dropped or
            # deferred, so only part of the gap passes through (capped at +40%).
            gap = target / have - 1 if target else 0.0
            self.desk_load[desk] = min(1.4, 1 + self.vacancy_passthrough * gap)


class WorkplaceStressModel(mesa.Model):
    def __init__(
        self,
        n_banks: int = 5,
        employees_per_bank: int = 120,
        contagion: float = 0.08,
        recovery_rate: float = 0.12,
        wellbeing_support: float = 0.0,
        hours_cap: float | None = None,
        base_error_rate: float = 0.004,
        base_quit_rate: float = 0.002,
        hiring_lag: int = 8,
        base_vol: float = 0.02,
        vol_persistence: float = 0.85,
        vol_feedback: float = 0.08,
        bank_beta: float = 1.1,
        shock_week: int | None = None,
        shock_size: float = 0.0,
        collect_agents: bool = True,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        self.contagion = contagion
        self.recovery_rate = recovery_rate
        self.wellbeing_support = wellbeing_support
        self.hours_cap = hours_cap
        self.base_error_rate = base_error_rate
        self.base_quit_rate = base_quit_rate
        self.hiring_lag = hiring_lag
        self.base_vol = base_vol
        self.vol_persistence = vol_persistence
        self.vol_feedback = vol_feedback
        self.bank_beta = bank_beta
        self.shock_week = shock_week
        self.shock_size = shock_size

        self.market_vol = base_vol
        self.market_ret = 0.0
        self.sector_ret = 0.0
        self.market_index = 100.0
        self.sector_returns: list[float] = []

        self.network = nx.Graph()
        self.employees_by_id: dict[int, Employee] = {}
        self._exits: list[Employee] = []
        self._hiring_queue: list[tuple[int, Bank, str, str]] = []
        self.exits_this_week = 0
        self.total_exits = 0
        self.peak_burnout = 0.0
        self.cum_op_losses = 0.0
        self.index_history = [self.market_index]

        self.banks = [self._build_bank(i, employees_per_bank) for i in range(n_banks)]

        model_reporters = {
            "MeanStress": lambda m: float(np.mean([e.stress for e in m.employees_by_id.values()])),
            "Healthy": lambda m: m.state_share(StressState.HEALTHY),
            "Stressed": lambda m: m.state_share(StressState.STRESSED),
            "Burnout": lambda m: m.state_share(StressState.BURNOUT),
            "MeanHours": lambda m: float(np.mean([e.hours for e in m.employees_by_id.values()])),
            "Headcount": lambda m: len(m.employees_by_id),
            "Exits": lambda m: m.exits_this_week,
            "OpLosses": lambda m: sum(b.op_losses for b in m.banks),
            "Productivity": lambda m: m.productivity(),
            "MarketVol": lambda m: m.market_vol,
            "MarketIndex": lambda m: m.market_index,
            "SectorReturn": lambda m: m.sector_ret,
            "FirmVol": lambda m: m.rolling_firm_vol(),
            "FirmIdioVol": lambda m: m.rolling_firm_vol(idio=True),
            "MacroVol": lambda m: m.rolling_macro_vol(),
            # Whole-run summaries (used by batch experiments).
            "FirmVolFull": lambda m: m.rolling_firm_vol(window=None),
            "FirmIdioVolFull": lambda m: m.rolling_firm_vol(window=None, idio=True),
            "MacroVolFull": lambda m: m.rolling_macro_vol(window=None),
            "TotalExits": lambda m: m.total_exits,
            "PeakBurnout": lambda m: m.peak_burnout,
            "CumOpLosses": lambda m: m.cum_op_losses,
            "MaxDrawdown": lambda m: m.max_drawdown(),
        }
        for b in self.banks:
            model_reporters[f"Bank{b.bank_id}_Return"] = (lambda m, b=b: b.ret)
        agent_reporters = (
            {
                "Stress": "stress",
                "State": lambda a: a.state.value,
                "Hours": "hours",
                "Role": "role",
                "Desk": "desk",
                "Bank": lambda a: a.bank.bank_id,
            }
            if collect_agents
            else None
        )
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )
        self.datacollector.collect(self)

    # --------------------------------------------------------------- building
    def _build_bank(self, bank_id: int, n: int) -> Bank:
        rng = self.rng
        bank = Bank(bank_id, n, capital=0.0)
        roles = list(ROLES)
        role_p = np.array([ROLES[r]["share"] for r in roles])
        desks = list(DESKS)

        for _ in range(n):
            role = rng.choice(roles, p=role_p / role_p.sum())
            desk = rng.choice(desks)
            e = Employee(self, bank, str(role), str(desk), tenure_weeks=int(rng.integers(26, 520)))
            self._register(e)
            bank.target[(e.desk, e.role)] += 1
            bank.desk_target[e.desk] += 1
            bank.ideal_revenue += ROLES[e.role]["value"]

        # Capital ~ 9 months of ideal revenue, so workforce-driven P&L swings
        # register in equity returns alongside market beta.
        bank.capital = bank.ideal_revenue * 52 * 0.75

        # Office network: small world inside each desk, reporting lines up the
        # ladder, and a few cross-desk friendships.
        by_desk = defaultdict(list)
        for e in bank.employees:
            by_desk[e.desk].append(e)
        for members in by_desk.values():
            k = min(6, len(members) - 1 - (len(members) - 1) % 2)
            if k >= 2:
                ws = nx.watts_strogatz_graph(len(members), k, 0.1, seed=int(rng.integers(1 << 31)))
                for u, v in ws.edges:
                    self.network.add_edge(members[u].unique_id, members[v].unique_id)
            for e in members:
                boss = self._pick_manager(e, members)
                if boss is not None:
                    self.network.add_edge(e.unique_id, boss.unique_id)
        for _ in range(n // 4):
            a, b = rng.choice(bank.employees, size=2, replace=False)
            self.network.add_edge(a.unique_id, b.unique_id)
        return bank

    def _pick_manager(self, e: Employee, members: list[Employee]) -> Employee | None:
        ladder = list(ROLES)
        idx = ladder.index(e.role)
        seniors = [m for m in members if ladder.index(m.role) > idx]
        return self.random.choice(seniors) if seniors else None

    def _register(self, e: Employee):
        self.employees_by_id[e.unique_id] = e
        e.bank.employees.append(e)
        self.network.add_node(e.unique_id)

    # ------------------------------------------------------------- lifecycle
    def schedule_exit(self, e: Employee):
        self._exits.append(e)

    def _process_exits(self):
        self.exits_this_week = len(self._exits)
        self.total_exits += self.exits_this_week
        for e in self._exits:
            e.bank.employees.remove(e)
            del self.employees_by_id[e.unique_id]
            self.network.remove_node(e.unique_id)
            self._hiring_queue.append((self.steps + self.hiring_lag, e.bank, e.role, e.desk))
            e.remove()
        self._exits.clear()

    def _process_hires(self):
        due = [h for h in self._hiring_queue if h[0] <= self.steps]
        self._hiring_queue = [h for h in self._hiring_queue if h[0] > self.steps]
        for _, bank, role, desk in due:
            e = Employee(self, bank, role, desk, tenure_weeks=0)
            e.stress = float(self.rng.beta(2, 6))
            colleagues = [c for c in bank.employees if c.desk == desk]
            self._register(e)
            if colleagues:
                for c in self.rng.choice(colleagues, size=min(4, len(colleagues)), replace=False):
                    self.network.add_edge(e.unique_id, c.unique_id)
                boss = self._pick_manager(e, colleagues)
                if boss is not None:
                    self.network.add_edge(e.unique_id, boss.unique_id)

    # ----------------------------------------------------------------- market
    def _update_market(self):
        """GARCH-like volatility with feedback from last week's bank losses."""
        rng = self.rng
        sector_loss = max(0.0, -self.sector_ret)
        self.market_vol = (
            self.base_vol
            + self.vol_persistence * (self.market_vol - self.base_vol)
            + self.vol_feedback * sector_loss
        )
        if self.shock_week is not None and self.steps == self.shock_week:
            self.market_vol += self.shock_size
        self.market_vol = float(np.clip(self.market_vol, 0.5 * self.base_vol, 0.15))
        self.market_ret = float(rng.normal(0.0015, self.market_vol))

    def _settle_banks(self):
        for b in self.banks:
            revenue = sum(e.output for e in b.employees)
            b.op_losses = sum(e.op_loss for e in b.employees)
            # Comp and fixed costs are sticky: they don't fall when output does.
            b.pnl = revenue - b.cost_ratio * b.ideal_revenue - b.op_losses
            idio = b.pnl / b.capital
            b.ret = self.bank_beta * self.market_ret + idio
            b.returns.append(b.ret)
            b.idio_returns.append(idio)
            b.update_desk_load()
        # The macro/sector index blends the broad market with bank equities.
        self.sector_ret = 0.6 * self.market_ret + 0.4 * float(np.mean([b.ret for b in self.banks]))
        self.sector_returns.append(self.sector_ret)
        self.market_index *= 1 + self.sector_ret
        self.index_history.append(self.market_index)
        self.cum_op_losses += sum(b.op_losses for b in self.banks)

    # ---------------------------------------------------------------- metrics
    def state_share(self, state: StressState) -> float:
        emps = self.employees_by_id.values()
        return sum(e.state is state for e in emps) / max(len(emps), 1)

    def productivity(self) -> float:
        ideal = sum(b.ideal_revenue for b in self.banks)
        actual = sum(e.output for e in self.employees_by_id.values())
        return actual / ideal if ideal else 0.0

    def rolling_firm_vol(self, window: int | None = 26, idio: bool = False) -> float:
        """Mean annualised volatility of bank returns over the last `window` weeks
        (``None`` = the whole run). ``idio`` strips out the market-beta component,
        leaving only the workforce-driven part of each bank's P&L."""
        if len(self.banks[0].returns) < 4:
            return float("nan")
        start = -window if window else 0
        series = [b.idio_returns if idio else b.returns for b in self.banks]
        return float(np.mean([np.std(r[start:]) for r in series]) * np.sqrt(52))

    def rolling_macro_vol(self, window: int | None = 26) -> float:
        if len(self.sector_returns) < 4:
            return float("nan")
        start = -window if window else 0
        return float(np.std(self.sector_returns[start:]) * np.sqrt(52))

    def max_drawdown(self) -> float:
        idx = np.asarray(self.index_history)
        return float(np.max(1 - idx / np.maximum.accumulate(idx)))

    # ------------------------------------------------------------------- step
    def step(self):
        self._update_market()
        self.agents.shuffle_do("step")
        self._settle_banks()
        self._process_exits()
        self._process_hires()
        self.peak_burnout = max(self.peak_burnout, self.state_share(StressState.BURNOUT))
        self.datacollector.collect(self)
