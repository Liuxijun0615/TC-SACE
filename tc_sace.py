"""Core TC-SACE-PM portfolio and stress-scenario co-evolution (requires NumPy).

Narratives do not enter the numeric objectives. The default narrative provider
uses fixed templates and does not require an external service.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Protocol

import numpy as np


TEMPLATES = ("MKT", "VOL", "LIQ", "MOM")
NORMALIZATION_EPSILON = 1e-12
FALLBACK_NARRATIVES = {
    "MKT": "A broad market sell-off places downward pressure on all stocks.",
    "VOL": "A rise in risk aversion places concentrated downward pressure on stocks with high recent volatility.",
    "LIQ": "A contraction in market liquidity places concentrated selling pressure on stocks with low recent liquidity.",
    "MOM": "A reversal in recent price trends causes stocks with high recent momentum to experience downward pressure.",
}


@dataclass(frozen=True)
class Config:
    universe_size: int = 100
    window: int = 60
    portfolio_size: int = 100
    scenario_size: int = 40
    generations: int = 200
    portfolio_archive_size: int = 100
    scenario_archive_size: int = 40
    max_weight: float = 0.1
    alpha_min: float = 0.25
    alpha_max: float = 2.0
    beta: float = 0.95
    elite_fraction: float = 0.2
    weight_mutation_sigma: float = 0.02
    alpha_mutation_sigma: float = 0.2
    template_mutation_probability: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if self.universe_size < 2 or self.window < 2:
            raise ValueError("universe_size and window must be at least 2")
        if not 0 < self.max_weight <= 1 or self.universe_size * self.max_weight < 1:
            raise ValueError("universe_size * max_weight must be at least 1")
        if self.portfolio_size < 2 or self.scenario_size < 4 or self.scenario_size % 4:
            raise ValueError("population sizes must be >=2; scenario_size must be divisible by 4")
        if self.generations < 1 or self.portfolio_archive_size < 1 or self.scenario_archive_size < 1:
            raise ValueError("generations and archive sizes must be positive")
        if not 0 < self.alpha_min < self.alpha_max or not 0 < self.beta < 1:
            raise ValueError("invalid alpha or CVaR confidence range")
        if not 0 < self.elite_fraction <= 1 or not 0 <= self.template_mutation_probability <= 1:
            raise ValueError("invalid elite fraction or template mutation probability")


@dataclass(frozen=True)
class Scenario:
    template: str
    alpha: float
    narrative: str

    def __post_init__(self) -> None:
        if self.template not in TEMPLATES or not np.isfinite(self.alpha):
            raise ValueError("invalid scenario core")


class NarrativeProvider(Protocol):
    """Optional LLM adapter: these methods never receive or change numeric fitness."""

    def initial(self, template: str) -> str: ...
    def rewrite(self, parent_a: str, parent_b: str, template: str) -> str: ...
    def consistent(self, narrative: str, template: str) -> bool: ...
    def repair(self, narrative: str, template: str) -> str: ...


class FallbackNarrativeProvider:
    def initial(self, template: str) -> str:
        return FALLBACK_NARRATIVES[template]

    def rewrite(self, parent_a: str, parent_b: str, template: str) -> str:
        return FALLBACK_NARRATIVES[template]

    def consistent(self, narrative: str, template: str) -> bool:
        return narrative == FALLBACK_NARRATIVES[template]

    def repair(self, narrative: str, template: str) -> str:
        return FALLBACK_NARRATIVES[template]


def safe_narrative(provider: NarrativeProvider, template: str, candidate: str | None = None) -> str:
    """Check, repair once, then use the supplement's fixed fallback."""
    try:
        proposed = provider.initial(template) if candidate is None else candidate
        if not isinstance(proposed, str) or not proposed.strip():
            raise ValueError("empty narrative")
        if provider.consistent(proposed, template):
            return proposed
        repaired = provider.repair(proposed, template)
        if isinstance(repaired, str) and repaired.strip() and provider.consistent(repaired, template):
            return repaired
    except (Exception):  # An explanation service must not stop numeric evolution.
        pass
    return FALLBACK_NARRATIVES[template]


def project_weights(values: np.ndarray, cap: float = 0.1) -> np.ndarray:
    """Euclidean projection onto {w: sum(w)=1, 0<=w<=cap}."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or cap <= 0 or len(x) * cap < 1:
        raise ValueError("invalid vector or infeasible cap")
    low, high = float(np.min(x) - cap), float(np.max(x))
    for _ in range(80):
        middle = (low + high) / 2
        total = np.clip(x - middle, 0, cap).sum()
        if total > 1:
            low = middle
        else:
            high = middle
    result = np.clip(x - (low + high) / 2, 0, cap)
    # Bisection error is below floating-point tolerance for the tested sizes.
    return result


@dataclass(frozen=True)
class MarketPanel:
    dates: tuple[str, ...]
    symbols: tuple[str, ...]
    close: np.ndarray
    volume: np.ndarray
    train_end: int  # Exclusive index into dates; no later row is used for fitting.

    @classmethod
    def select_universe(
        cls, dates: list[str] | tuple[str, ...], symbols: list[str] | tuple[str, ...],
        close: np.ndarray, volume: np.ndarray, train_end: int, config: Config,
    ) -> MarketPanel:
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        if close.shape != volume.shape or close.ndim != 2 or close.shape != (len(dates), len(symbols)):
            raise ValueError("close and volume must be T x N arrays matching dates and symbols")
        if len(set(dates)) != len(dates) or list(dates) != sorted(dates) or len(set(symbols)) != len(symbols):
            raise ValueError("dates must be sorted and unique; symbols must be unique")
        if not config.window + 2 <= train_end <= len(dates):
            raise ValueError("training period must contain at least window+2 dates")
        if len(symbols) < config.universe_size:
            raise ValueError("too few candidate symbols")
        training_complete = np.all(np.isfinite(close[:train_end]) & (close[:train_end] > 0)
                                   & np.isfinite(volume[:train_end]) & (volume[:train_end] >= 0), axis=0)
        if training_complete.sum() < config.universe_size:
            raise ValueError("too few stocks with complete training OHLCV data")
        turnover = np.mean(close[:train_end] * volume[:train_end], axis=0)
        indices = sorted(np.flatnonzero(training_complete), key=lambda i: (-turnover[i], str(symbols[i])))
        chosen = indices[:config.universe_size]
        selected_close = close[:, chosen].copy()
        selected_volume = volume[:, chosen].copy()
        if not np.all(np.isfinite(selected_close) & (selected_close > 0)):
            raise ValueError("selected symbols have missing or invalid close prices")
        if not np.all(np.isfinite(selected_volume) & (selected_volume >= 0)):
            raise ValueError("selected symbols have missing or invalid volume")
        return cls(tuple(dates), tuple(str(symbols[i]) for i in chosen), selected_close,
                   selected_volume, train_end)

    def window_data(self, start: int, end: int, window: int) -> EvaluationData:
        if start < window + 1 or start >= end or end > len(self.dates):
            raise ValueError("evaluation range needs prior window returns and at least one day")
        returns_all = np.empty_like(self.close)
        returns_all[0] = np.nan
        returns_all[1:] = self.close[1:] / self.close[:-1] - 1
        times = np.arange(start, end)
        history = np.stack([returns_all[t - window:t] for t in times])
        prior_turnover = np.stack([(self.close * self.volume)[t - window:t] for t in times])
        sigma = history.std(axis=1, ddof=1)
        liquidity = prior_turnover.mean(axis=1)
        momentum = history.sum(axis=1)
        return EvaluationData(tuple(self.dates[start:end]), returns_all[start:end], sigma,
                              liquidity, momentum)

    def training_data(self, window: int) -> EvaluationData:
        return self.window_data(window + 1, self.train_end, window)

    def holdout_data(self, window: int) -> EvaluationData:
        return self.window_data(self.train_end, len(self.dates), window)


@dataclass(frozen=True)
class EvaluationData:
    dates: tuple[str, ...]
    returns: np.ndarray
    sigma: np.ndarray
    liquidity: np.ndarray
    momentum: np.ndarray
    event_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = self.returns.shape
        if len(shape) != 2 or not all(x.shape == shape for x in (self.sigma, self.liquidity, self.momentum)):
            raise ValueError("evaluation arrays must have the same T x N shape")
        if shape[0] < 1 or not all(np.all(np.isfinite(x)) for x in
                                  (self.returns, self.sigma, self.liquidity, self.momentum)):
            raise ValueError("evaluation arrays must be nonempty and finite")
        if np.any(self.returns <= -1) or np.any(self.sigma < 0):
            raise ValueError("invalid daily returns or volatility")
        if self.event_mask is not None:
            if self.event_mask.shape != (shape[0],) or self.event_mask.dtype != np.bool_:
                raise ValueError("event mask must be a boolean vector matching evaluation dates")

    def masks(self, target_fraction: float = 0.3) -> dict[str, np.ndarray]:
        if not 0 < target_fraction < 1:
            raise ValueError("target fraction must be between zero and one")
        return {
            "MKT": np.ones_like(self.returns, dtype=bool),
            "VOL": self.sigma >= np.quantile(self.sigma, 1 - target_fraction, axis=1)[:, None],
            "LIQ": self.liquidity <= np.quantile(self.liquidity, target_fraction, axis=1)[:, None],
            "MOM": self.momentum >= np.quantile(self.momentum, 1 - target_fraction, axis=1)[:, None],
        }

    def stressed_returns(self, scenario: Scenario, masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
        mask = (self.masks() if masks is None else masks)[scenario.template]
        targeted_sigma = mask * self.sigma
        full_norm = np.linalg.norm(self.sigma, axis=1)
        targeted_norm = np.linalg.norm(targeted_sigma, axis=1)
        gamma = full_norm / np.maximum(targeted_norm, NORMALIZATION_EPSILON)
        event = 1 if self.event_mask is None else self.event_mask[:, None]
        stressed = self.returns - event * scenario.alpha * gamma[:, None] * targeted_sigma
        if np.any(stressed <= -1):
            raise ValueError("stress return <= -100%; paper's unclipped compounding is undefined")
        return stressed


def return_metrics(daily: np.ndarray, beta: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative return and manuscript CVaR, with quantile-inclusive tail."""
    daily = np.asarray(daily, dtype=float)
    if daily.ndim < 1 or daily.shape[-1] < 1 or np.any(daily <= -1):
        raise ValueError("daily returns must be nonempty and greater than -1")
    cr = np.expm1(np.log1p(daily).sum(axis=-1))
    losses = -daily
    var = np.quantile(losses, beta, axis=-1, keepdims=True)
    tail = losses >= var
    cvar = np.sum(np.where(tail, losses, 0), axis=-1) / np.sum(tail, axis=-1)
    return cr, cvar


@dataclass(frozen=True)
class PortfolioEvaluation:
    objectives: np.ndarray
    cumulative_return: np.ndarray
    cvar: np.ndarray
    adversarial_risk: np.ndarray
    stress_losses: np.ndarray  # scenario x portfolio


def evaluate_portfolios(weights: np.ndarray, scenarios: list[Scenario], data: EvaluationData,
                        target_fraction: float = 0.3) -> PortfolioEvaluation:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != data.returns.shape[1] or len(scenarios) < 1:
        raise ValueError("weights must be P x N and at least one scenario is required")
    normal_daily = weights @ data.returns.T
    cr, cvar = return_metrics(normal_daily)
    masks = data.masks(target_fraction)
    stressed = np.stack([data.stressed_returns(s, masks) for s in scenarios])
    stressed_daily = np.einsum("stn,pn->spt", stressed, weights, optimize=True)
    stressed_cr, stressed_cvar = return_metrics(stressed_daily)
    stress_losses = -stressed_cr + stressed_cvar
    adv = stress_losses.max(axis=0)
    return PortfolioEvaluation(np.column_stack((-cr, cvar, adv)), cr, cvar, adv, stress_losses)


def scenario_distance(a: Scenario, b: Scenario, config: Config) -> float:
    return float(a.template != b.template) + abs(a.alpha - b.alpha) / (config.alpha_max - config.alpha_min)


def evaluate_scenarios(
    scenarios: list[Scenario], stress_losses: np.ndarray, base_loss: np.ndarray,
    elite_indices: np.ndarray, archive: list[Scenario], config: Config,
    exclude_same_object: bool = False,
) -> np.ndarray:
    if stress_losses.shape != (len(scenarios), len(base_loss)):
        raise ValueError("stress loss shape does not match populations")
    damage = np.maximum(0, stress_losses[:, elite_indices] - base_loss[elite_indices]).mean(axis=1)
    cost = np.array([(s.alpha - config.alpha_min) / (config.alpha_max - config.alpha_min) for s in scenarios])
    diversity = np.array([
        min((scenario_distance(s, old, config) for old in archive
             if not (exclude_same_object and s is old)), default=1.0)
        for s in scenarios
    ])
    return np.column_stack((-damage, cost, -diversity))


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("objectives must be finite two-dimensional array")
    # Pairwise dominance is small (at most 2*N_P) and much faster in NumPy
    # than repeated Python comparisons over every generation.
    weak = np.all(values[:, None, :] <= values[None, :, :], axis=2)
    strict = np.any(values[:, None, :] < values[None, :, :], axis=2)
    dominance = weak & strict
    counts = dominance.sum(axis=0).astype(int)
    assigned = np.zeros(len(values), dtype=bool)
    fronts: list[list[int]] = []
    front = np.flatnonzero(counts == 0).tolist()
    while front:
        fronts.append(front)
        assigned[front] = True
        counts -= dominance[front].sum(axis=0)
        front = np.flatnonzero((counts == 0) & ~assigned).tolist()
    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> dict[int, float]:
    values = np.asarray(objectives, dtype=float)
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for objective in range(values.shape[1]):
        ordered = sorted(front, key=lambda i: (values[i, objective], i))
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        span = values[ordered[-1], objective] - values[ordered[0], objective]
        if span > 0:
            for pos in range(1, len(ordered) - 1):
                distance[ordered[pos]] += (values[ordered[pos + 1], objective]
                                           - values[ordered[pos - 1], objective]) / span
    return distance


def nsga_order(objectives: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
    fronts = non_dominated_sort(objectives)
    ranks = np.empty(len(objectives), dtype=int)
    crowding = np.empty(len(objectives), dtype=float)
    order = []
    for rank, front in enumerate(fronts):
        distances = crowding_distance(objectives, front)
        for i in front:
            ranks[i] = rank
            crowding[i] = distances[i]
        order.extend(sorted(front, key=lambda i: (-distances[i], i)))
    return order, ranks, crowding


def tournament(ranks: np.ndarray, crowding: np.ndarray, rng: np.random.Generator) -> int:
    a, b = rng.choice(len(ranks), size=2, replace=False)
    if ranks[a] != ranks[b]:
        return int(a if ranks[a] < ranks[b] else b)
    if crowding[a] != crowding[b]:
        return int(a if crowding[a] > crowding[b] else b)
    return int(a if rng.random() < 0.5 else b)


def balanced_index(objectives: np.ndarray) -> int:
    """Reproducible, training-only representative of the Pareto archive."""
    low, high = objectives.min(axis=0), objectives.max(axis=0)
    scale = np.where(high > low, high - low, 1)
    distances = np.linalg.norm((objectives - low) / scale, axis=1)
    return int(np.argmin(distances))


@dataclass
class EvolutionResult:
    portfolio_population: np.ndarray
    scenario_population: list[Scenario]
    portfolio_archive: np.ndarray
    portfolio_archive_objectives: np.ndarray
    scenario_archive: list[Scenario]
    scenario_archive_objectives: np.ndarray
    history: list[dict[str, float | int]]
    initial_portfolio_population: np.ndarray | None = None
    initial_scenario_population: list[Scenario] | None = None
    portfolio_evaluations: int = 0
    scenario_evaluations: int = 0

    def balanced_weights(self) -> np.ndarray:
        return self.portfolio_archive[balanced_index(self.portfolio_archive_objectives)].copy()


class TCSACE:
    def __init__(self, config: Config, provider: NarrativeProvider | None = None):
        self.config = config
        self.provider = provider or FallbackNarrativeProvider()
        self.rng = np.random.default_rng(config.seed)

    def _initial_scenarios(self) -> list[Scenario]:
        templates = list(TEMPLATES) * (self.config.scenario_size // 4)
        self.rng.shuffle(templates)
        return [Scenario(t, float(self.rng.uniform(self.config.alpha_min, self.config.alpha_max)),
                         safe_narrative(self.provider, t)) for t in templates]

    def _portfolio_children(self, weights: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        _, ranks, crowding = nsga_order(objectives)
        children = []
        for _ in range(self.config.portfolio_size):
            a = weights[tournament(ranks, crowding, self.rng)]
            b = weights[tournament(ranks, crowding, self.rng)]
            lam = self.rng.random()
            child = lam * a + (1 - lam) * b
            child += self.rng.normal(0, self.config.weight_mutation_sigma, size=len(child))
            children.append(project_weights(child, self.config.max_weight))
        return np.stack(children)

    def _scenario_children(self, scenarios: list[Scenario], objectives: np.ndarray) -> list[Scenario]:
        _, ranks, crowding = nsga_order(objectives)
        children = []
        for _ in range(self.config.scenario_size):
            a = scenarios[tournament(ranks, crowding, self.rng)]
            b = scenarios[tournament(ranks, crowding, self.rng)]
            inherited = a.template if self.rng.random() < 0.5 else b.template
            lam = self.rng.random()
            alpha = float(np.clip(lam * a.alpha + (1 - lam) * b.alpha
                                  + self.rng.normal(0, self.config.alpha_mutation_sigma),
                                  self.config.alpha_min, self.config.alpha_max))
            template = inherited
            if self.rng.random() < self.config.template_mutation_probability:
                template = str(self.rng.choice([t for t in TEMPLATES if t != inherited]))
            try:
                candidate = self.provider.rewrite(a.narrative, b.narrative, inherited)
            except (Exception):
                candidate = ""
            children.append(Scenario(template, alpha, safe_narrative(self.provider, template, candidate)))
        return children

    def fit(self, data: EvaluationData,
            progress: Callable[[dict[str, float | int]], None] | None = None) -> EvolutionResult:
        cfg = self.config
        if data.returns.shape[1] != cfg.universe_size:
            raise ValueError("evaluation data does not match selected universe size")
        weights = np.stack([project_weights(self.rng.random(cfg.universe_size), cfg.max_weight)
                            for _ in range(cfg.portfolio_size)])
        initial_weights = weights.copy()
        scenarios = self._initial_scenarios()
        initial_scenarios = list(scenarios)
        portfolio_archive = np.empty((0, cfg.universe_size))
        scenario_archive: list[Scenario] = []
        history: list[dict[str, float | int]] = []
        portfolio_evaluations = 0
        scenario_evaluations = 0
        for generation in range(cfg.generations):
            parents_p = evaluate_portfolios(weights, scenarios, data)
            portfolio_evaluations += len(weights) * len(scenarios)
            elite_count = ceil(cfg.elite_fraction * cfg.portfolio_size)
            elite = np.array(nsga_order(parents_p.objectives)[0][:elite_count])
            feedback_weights = weights
            base_loss = -parents_p.cumulative_return + parents_p.cvar
            parents_s_obj = evaluate_scenarios(scenarios, parents_p.stress_losses,
                                               base_loss, elite, scenario_archive, cfg)
            scenario_evaluations += len(scenarios) * len(feedback_weights)

            children_p = self._portfolio_children(weights, parents_p.objectives)
            children_p_eval = evaluate_portfolios(children_p, scenarios, data)
            portfolio_evaluations += len(children_p) * len(scenarios)
            all_p = np.concatenate((weights, children_p))
            all_p_obj = np.concatenate((parents_p.objectives, children_p_eval.objectives))
            selected_p = nsga_order(all_p_obj)[0][:cfg.portfolio_size]
            weights = all_p[selected_p].copy()

            children_s = self._scenario_children(scenarios, parents_s_obj)
            children_s_losses = evaluate_portfolios(feedback_weights, children_s, data).stress_losses
            portfolio_evaluations += len(feedback_weights) * len(children_s)
            children_s_obj = evaluate_scenarios(children_s, children_s_losses, base_loss,
                                                elite, scenario_archive, cfg)
            scenario_evaluations += len(children_s) * len(feedback_weights)
            all_s = scenarios + children_s
            all_s_obj = np.concatenate((parents_s_obj, children_s_obj))
            selected_s = nsga_order(all_s_obj)[0][:cfg.scenario_size]
            scenarios = [all_s[i] for i in selected_s]

            # Every archive candidate is scored against the same new opposing
            # population.  Objectives from the previous generation are stale.
            candidate_p = np.concatenate((portfolio_archive, weights))
            candidate_p_obj = evaluate_portfolios(candidate_p, scenarios, data).objectives
            portfolio_evaluations += len(candidate_p) * len(scenarios)
            front_p = non_dominated_sort(candidate_p_obj)[0]
            distances_p = crowding_distance(candidate_p_obj, front_p)
            keep_p = sorted(front_p, key=lambda i: (-distances_p[i], i))[:cfg.portfolio_archive_size]
            portfolio_archive = candidate_p[keep_p].copy()
            portfolio_archive_obj = candidate_p_obj[keep_p].copy()

            candidate_s = scenario_archive + scenarios
            next_feedback_eval = evaluate_portfolios(weights, candidate_s, data)
            portfolio_evaluations += len(weights) * len(candidate_s)
            next_elite = np.array(nsga_order(evaluate_portfolios(weights, scenarios, data).objectives)[0][:elite_count])
            portfolio_evaluations += len(weights) * len(scenarios)
            next_base = -next_feedback_eval.cumulative_return + next_feedback_eval.cvar
            candidate_s_obj = evaluate_scenarios(candidate_s, next_feedback_eval.stress_losses,
                                                 next_base, next_elite, scenario_archive, cfg,
                                                 exclude_same_object=True)
            scenario_evaluations += len(candidate_s) * len(weights)
            front_s = non_dominated_sort(candidate_s_obj)[0]
            distances = crowding_distance(candidate_s_obj, front_s)
            keep_s = sorted(front_s, key=lambda i: (-distances[i], i))[:cfg.scenario_archive_size]
            scenario_archive = [candidate_s[i] for i in keep_s]
            scenario_archive_obj = candidate_s_obj[keep_s].copy()
            history.append({
                "generation": generation + 1,
                "best_cr": float((-all_p_obj[selected_p, 0]).max()),
                "lowest_cvar": float(all_p_obj[selected_p, 1].min()),
                "lowest_adv_risk": float(all_p_obj[selected_p, 2].min()),
                "highest_damage": float((-all_s_obj[selected_s, 0]).max()),
                "portfolio_archive_size": len(portfolio_archive),
                "scenario_archive_size": len(scenario_archive),
            })
            if progress is not None:
                progress(history[-1])
        return EvolutionResult(weights, scenarios, portfolio_archive, portfolio_archive_obj,
                               scenario_archive, scenario_archive_obj, history,
                               initial_weights, initial_scenarios,
                               portfolio_evaluations, scenario_evaluations)
