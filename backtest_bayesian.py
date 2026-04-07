from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from Dataset import StockDataset
from backtest_baseline import (
    DailyPortfolioTradingStrategy,
    build_evaluation_split,
    generate_close_prices,
    get_device,
    grid_search_strategy,
    load_file,
    load_model,
    make_json_safe,
    normalize_split,
    parse_grid,
    predict_probabilities,
    print_metrics,
    split_data_by_time,
)


def predict_probabilities_mc(
    model: torch.nn.Module,
    split_data: torch.Tensor,
    lookback: int,
    device: torch.device,
    num_mc_runs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = StockDataset(split_data, lookback, device)
    num_days = len(dataset)
    if num_days <= 0:
        raise ValueError("Split is too short for the requested lookback window.")

    sample_x, _ = dataset[0]
    n_stocks = sample_x.shape[0]

    model.train()
    mc_probabilities = []

    with torch.no_grad():
        for run_idx in tqdm(range(num_mc_runs), desc="MC Dropout inference"):
            one_run = torch.zeros(n_stocks, num_days, 2, device=device)
            for day_idx in range(num_days):
                x, _ = dataset[day_idx]
                logits, *_ = model(x)
                one_run[:, day_idx, :] = torch.softmax(logits, dim=-1)
            mc_probabilities.append(one_run)

            if (run_idx + 1) % 10 == 0 or run_idx + 1 == num_mc_runs:
                print(f"  finished MC run {run_idx + 1}/{num_mc_runs}")

    stacked = torch.stack(mc_probabilities, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)


class BayesianPortfolioTradingStrategy(DailyPortfolioTradingStrategy):
    def __init__(
        self,
        *args,
        prob_threshold: float = 0.5,
        variance_quantile: float = 0.8,
        variance_weight: float = 10.0,
        market_variance_threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not (0.0 < prob_threshold < 1.0):
            raise ValueError("prob_threshold must satisfy 0 < prob_threshold < 1.")
        if not (0.0 < variance_quantile <= 1.0):
            raise ValueError("variance_quantile must satisfy 0 < variance_quantile <= 1.")
        if variance_weight < 0.0:
            raise ValueError("variance_weight must be non-negative.")
        if market_variance_threshold is not None and market_variance_threshold < 0.0:
            raise ValueError("market_variance_threshold must be non-negative.")

        self.prob_threshold = prob_threshold
        self.variance_quantile = variance_quantile
        self.variance_weight = variance_weight
        self.market_variance_threshold = market_variance_threshold
        self.filtered_out_counts: List[int] = []
        self.skipped_days_due_to_market_uncertainty = 0
        self.daily_market_variances: List[float] = []

    def reset_tracking(self) -> None:
        super().reset_tracking()
        self.filtered_out_counts = []
        self.skipped_days_due_to_market_uncertainty = 0
        self.daily_market_variances = []

    def should_skip_day(
        self,
        mean_rise_probs: torch.Tensor,
        rise_variances: torch.Tensor,
    ) -> bool:
        candidate_mask = mean_rise_probs > self.prob_threshold
        if torch.any(candidate_mask):
            day_market_variance = rise_variances[candidate_mask].mean()
        else:
            day_market_variance = rise_variances.mean()

        self.daily_market_variances.append(float(day_market_variance))

        if self.market_variance_threshold is None:
            return False

        if float(day_market_variance) > self.market_variance_threshold:
            self.skipped_days_due_to_market_uncertainty += 1
            return True

        return False

    def select_stocks_with_uncertainty(
        self,
        mean_rise_probs: torch.Tensor,
        rise_variances: torch.Tensor,
    ) -> Tuple[List[int], int]:
        n_stocks = mean_rise_probs.shape[0]
        target_stocks = max(1, int(self.p_ratio * n_stocks))

        candidate_mask = mean_rise_probs > self.prob_threshold
        candidate_indices = torch.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            self.filtered_out_counts.append(0)
            return [], 0

        candidate_variances = rise_variances[candidate_indices]
        variance_cutoff = torch.quantile(candidate_variances, self.variance_quantile)
        filtered_mask = candidate_variances <= variance_cutoff
        filtered_candidates = candidate_indices[filtered_mask]
        self.filtered_out_counts.append(int(len(candidate_indices) - len(filtered_candidates)))

        m_rising = len(filtered_candidates)
        stop_loss_threshold = int(target_stocks * self.q_stop_loss)
        adjusted_scores = mean_rise_probs - self.variance_weight * rise_variances

        if m_rising >= target_stocks:
            selected_scores = adjusted_scores[filtered_candidates]
            top_indices = torch.topk(selected_scores, target_stocks).indices
            return filtered_candidates[top_indices].tolist(), target_stocks

        if m_rising >= stop_loss_threshold:
            num_to_select = int(m_rising * self.r_rising_ratio)
            if num_to_select <= 0:
                return [], 0
            selected_scores = adjusted_scores[filtered_candidates]
            top_indices = torch.topk(selected_scores, num_to_select).indices
            return filtered_candidates[top_indices].tolist(), num_to_select

        return [], 0

    def run_backtest(
        self,
        mean_rise_probabilities: torch.Tensor,
        close_prices: torch.Tensor,
        rise_variances: torch.Tensor,
    ) -> Dict[str, float]:
        self.reset_tracking()

        mean_probs_np = (
            mean_rise_probabilities.cpu().numpy()
            if isinstance(mean_rise_probabilities, torch.Tensor)
            else mean_rise_probabilities
        )
        variances_np = (
            rise_variances.cpu().numpy()
            if isinstance(rise_variances, torch.Tensor)
            else rise_variances
        )
        close_prices_np = (
            close_prices.cpu().numpy() if isinstance(close_prices, torch.Tensor) else close_prices
        )

        n_stocks, t_days = mean_probs_np.shape
        if close_prices_np.shape[1] != t_days + 1:
            raise ValueError("close_prices must have one more day than rise probabilities.")

        cash = self.initial_capital
        holdings: Dict[int, float] = {}
        self.portfolio_values.append(self.initial_capital)
        self.cash_history.append(cash)

        for day_idx in range(t_days):
            day_mean_probs = torch.tensor(mean_probs_np[:, day_idx])
            day_variances = torch.tensor(variances_np[:, day_idx])
            current_prices = close_prices_np[:, day_idx]

            if day_idx % self.rebalance_frequency == 0:
                if self.should_skip_day(day_mean_probs, day_variances):
                    self.filtered_out_counts.append(0)
                    target_stocks = []
                else:
                    target_stocks, _ = self.select_stocks_with_uncertainty(
                        day_mean_probs,
                        day_variances,
                    )
                holdings, cash, transaction_cost, num_trades = self.execute_trades(
                    holdings,
                    target_stocks,
                    current_prices,
                    cash,
                )
            else:
                transaction_cost = 0.0
                num_trades = 0

            self.transaction_costs.append(transaction_cost)
            self.transaction_counts.append(num_trades)
            self.daily_holdings.append(list(holdings.keys()))

            next_prices = close_prices_np[:, day_idx + 1]
            portfolio_value = cash
            for stock_idx, shares in holdings.items():
                portfolio_value += shares * next_prices[stock_idx]

            previous_value = self.portfolio_values[-1]
            daily_return = (portfolio_value - previous_value) / previous_value

            self.portfolio_values.append(portfolio_value)
            self.cash_history.append(cash)
            self.daily_returns.append(daily_return)
            self.cumulative_returns.append((portfolio_value / self.initial_capital) - 1.0)

        metrics = self.calculate_performance_metrics()
        metrics["avg_filtered_out"] = (
            float(sum(self.filtered_out_counts) / len(self.filtered_out_counts))
            if self.filtered_out_counts
            else 0.0
        )
        metrics["skipped_days_due_to_market_uncertainty"] = float(
            self.skipped_days_due_to_market_uncertainty
        )
        metrics["avg_market_variance"] = (
            float(sum(self.daily_market_variances) / len(self.daily_market_variances))
            if self.daily_market_variances
            else 0.0
        )
        return metrics


def grid_search_bayesian_strategy(
    mean_rise_probabilities: torch.Tensor,
    close_prices: torch.Tensor,
    rise_variances: torch.Tensor,
    p_grid: List[float],
    q_grid: List[float],
    r_grid: List[float],
    initial_capital: float,
    transaction_cost_rate: float,
    risk_free_rate: float,
    rebalance_frequency: int,
    prob_threshold: float,
    variance_quantile: float,
    variance_weight: float,
    market_variance_threshold: Optional[float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    best_config: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_objective = (-float("inf"), -float("inf"))

    for p_ratio in p_grid:
        for q_stop_loss in q_grid:
            for r_rising_ratio in r_grid:
                strategy = BayesianPortfolioTradingStrategy(
                    initial_capital=initial_capital,
                    transaction_cost_rate=transaction_cost_rate,
                    p_ratio=p_ratio,
                    q_stop_loss=q_stop_loss,
                    r_rising_ratio=r_rising_ratio,
                    risk_free_rate=risk_free_rate,
                    rebalance_frequency=rebalance_frequency,
                    prob_threshold=prob_threshold,
                    variance_quantile=variance_quantile,
                    variance_weight=variance_weight,
                    market_variance_threshold=market_variance_threshold,
                )
                metrics = strategy.run_backtest(
                    mean_rise_probabilities=mean_rise_probabilities,
                    close_prices=close_prices,
                    rise_variances=rise_variances,
                )
                objective = (metrics["sharpe_ratio"], metrics["annual_return"])
                if objective > best_objective:
                    best_objective = objective
                    best_config = {
                        "p_ratio": p_ratio,
                        "q_stop_loss": q_stop_loss,
                        "r_rising_ratio": r_rising_ratio,
                    }
                    best_metrics = metrics

    if best_config is None or best_metrics is None:
        raise RuntimeError("Bayesian grid search did not evaluate any configurations.")

    return best_config, best_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Bayesian backtests with MC Dropout uncertainty filtering.")
    parser.add_argument("--model-version", default="Magnetv1")
    parser.add_argument("--data-name", default="NASDAQ100")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--num-mc-runs", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-heads-mha", type=int, default=2)
    parser.add_argument("--num-channels", type=int, default=4)
    parser.add_argument("--num-heads-causal-mha", type=int, default=2)
    parser.add_argument("--num-mage", type=int, default=1)
    parser.add_argument("--num-f2dattn", type=int, default=1)
    parser.add_argument("--num-tch", type=int, default=2)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--m1", type=int, default=64)
    parser.add_argument("--num-s2dattn", type=int, default=1)
    parser.add_argument("--num-gph", type=int, default=2)
    parser.add_argument("--m2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--transaction-cost-rate", type=float, default=0.0025)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--rebalance-frequency", type=int, default=1)
    parser.add_argument("--fixed-p-ratio", type=float, default=None)
    parser.add_argument("--fixed-q-stop-loss", type=float, default=None)
    parser.add_argument("--fixed-r-rising-ratio", type=float, default=None)
    parser.add_argument("--deterministic-val", action="store_true")
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--variance-quantile", type=float, default=0.8)
    parser.add_argument("--variance-weight", type=float, default=10.0)
    parser.add_argument("--market-variance-threshold", type=float, default=None)
    parser.add_argument("--p-grid", type=float, nargs="*", default=[])
    parser.add_argument("--q-grid", type=float, nargs="*", default=[])
    parser.add_argument("--r-grid", type=float, nargs="*", default=[])
    parser.add_argument("--output-dir", default="backtest_outputs")
    parser.add_argument("--eval-split", choices=["test", "val_test"], default="test")
    return parser


def main(args: argparse.Namespace) -> None:
    device = get_device()
    print("Using device:", device)
    print(f"Using model: {args.model_version}")
    print("Prediction mode: mc")

    data = load_file(args.data_path, map_location=device).to(device)
    split_data = split_data_by_time(data)

    model = load_model(
        model_version=args.model_version,
        weight_path=args.weight_path,
        data_shape=tuple(data.shape),
        dim=args.dim,
        num_experts=args.num_experts,
        num_heads_mha=args.num_heads_mha,
        num_channels=args.num_channels,
        num_heads_causal_mha=args.num_heads_causal_mha,
        lookback=args.lookback,
        num_mage=args.num_mage,
        num_f2dattn=args.num_f2dattn,
        num_tch=args.num_tch,
        topk=args.topk,
        m1=args.m1,
        num_s2dattn=args.num_s2dattn,
        num_gph=args.num_gph,
        m2=args.m2,
        dropout=args.dropout,
        device=device,
    )

    mean_probs_by_split: Dict[str, torch.Tensor] = {}
    variances_by_split: Dict[str, Optional[torch.Tensor]] = {}
    close_prices_by_split: Dict[str, torch.Tensor] = {}

    eval_label = "Test Metrics" if args.eval_split == "test" else "Combined Val+Test Metrics"
    fixed_config = (
        args.fixed_p_ratio is not None
        and args.fixed_q_stop_loss is not None
        and args.fixed_r_rising_ratio is not None
    )
    needs_validation = (not fixed_config) or args.eval_split == "test"

    if needs_validation:
        print("\nPreparing val split...")
        val_raw_split = split_data["val"]
        val_normalized_split = normalize_split(val_raw_split)
        if args.deterministic_val:
            val_probs, _ = predict_probabilities(
                model=model,
                split_data=val_normalized_split,
                lookback=args.lookback,
                device=device,
            )
            mean_probs_by_split["val"] = val_probs
            variances_by_split["val"] = None
            print("Validation will use deterministic probabilities for tuning.")
        else:
            val_mean_probs, val_variances = predict_probabilities_mc(
                model=model,
                split_data=val_normalized_split,
                lookback=args.lookback,
                device=device,
                num_mc_runs=args.num_mc_runs,
            )
            mean_probs_by_split["val"] = val_mean_probs
            variances_by_split["val"] = val_variances
        close_prices_by_split["val"] = generate_close_prices(val_raw_split, args.lookback)

    print(f"\nPreparing {args.eval_split} evaluation split...")
    eval_raw_split = build_evaluation_split(split_data, args.eval_split)
    eval_normalized_split = normalize_split(eval_raw_split)
    eval_mean_probs, eval_variances = predict_probabilities_mc(
        model=model,
        split_data=eval_normalized_split,
        lookback=args.lookback,
        device=device,
        num_mc_runs=args.num_mc_runs,
    )
    mean_probs_by_split["eval"] = eval_mean_probs
    variances_by_split["eval"] = eval_variances
    close_prices_by_split["eval"] = generate_close_prices(eval_raw_split, args.lookback)

    if fixed_config:
        best_config = {
            "p_ratio": args.fixed_p_ratio,
            "q_stop_loss": args.fixed_q_stop_loss,
            "r_rising_ratio": args.fixed_r_rising_ratio,
        }
        print("\nUsing fixed config:", best_config)
        val_metrics = None
        if args.eval_split == "test":
            if args.deterministic_val:
                val_strategy = DailyPortfolioTradingStrategy(
                    initial_capital=args.initial_capital,
                    transaction_cost_rate=args.transaction_cost_rate,
                    p_ratio=best_config["p_ratio"],
                    q_stop_loss=best_config["q_stop_loss"],
                    r_rising_ratio=best_config["r_rising_ratio"],
                    risk_free_rate=args.risk_free_rate,
                    rebalance_frequency=args.rebalance_frequency,
                )
                val_metrics = val_strategy.run_backtest(
                    mean_probs_by_split["val"][:, :, 1],
                    close_prices_by_split["val"],
                )
            else:
                val_strategy = BayesianPortfolioTradingStrategy(
                    initial_capital=args.initial_capital,
                    transaction_cost_rate=args.transaction_cost_rate,
                    p_ratio=best_config["p_ratio"],
                    q_stop_loss=best_config["q_stop_loss"],
                    r_rising_ratio=best_config["r_rising_ratio"],
                    risk_free_rate=args.risk_free_rate,
                    rebalance_frequency=args.rebalance_frequency,
                    prob_threshold=args.prob_threshold,
                    variance_quantile=args.variance_quantile,
                    variance_weight=args.variance_weight,
                    market_variance_threshold=args.market_variance_threshold,
                )
                val_metrics = val_strategy.run_backtest(
                    mean_rise_probabilities=mean_probs_by_split["val"][:, :, 1],
                    close_prices=close_prices_by_split["val"],
                    rise_variances=variances_by_split["val"][:, :, 1],
                )
    else:
        p_grid = parse_grid(args.p_grid, 0.05, 1.0, 0.05)
        q_grid = parse_grid(args.q_grid, 0.05, 0.95, 0.05)
        r_grid = parse_grid(args.r_grid, 0.0, 1.0, 0.05)

        if args.deterministic_val:
            best_config, val_metrics = grid_search_strategy(
                rise_probabilities=mean_probs_by_split["val"][:, :, 1],
                close_prices=close_prices_by_split["val"],
                p_grid=p_grid,
                q_grid=q_grid,
                r_grid=r_grid,
                initial_capital=args.initial_capital,
                transaction_cost_rate=args.transaction_cost_rate,
                risk_free_rate=args.risk_free_rate,
            )
            print("\nBest validation config from deterministic validation:", best_config)
        else:
            best_config, val_metrics = grid_search_bayesian_strategy(
                mean_rise_probabilities=mean_probs_by_split["val"][:, :, 1],
                close_prices=close_prices_by_split["val"],
                rise_variances=variances_by_split["val"][:, :, 1],
                p_grid=p_grid,
                q_grid=q_grid,
                r_grid=r_grid,
                initial_capital=args.initial_capital,
                transaction_cost_rate=args.transaction_cost_rate,
                risk_free_rate=args.risk_free_rate,
                rebalance_frequency=args.rebalance_frequency,
                prob_threshold=args.prob_threshold,
                variance_quantile=args.variance_quantile,
                variance_weight=args.variance_weight,
                market_variance_threshold=args.market_variance_threshold,
            )
            print("\nBest validation config:", best_config)

    if val_metrics is not None:
        print_metrics("Validation Metrics", val_metrics)

    test_strategy = BayesianPortfolioTradingStrategy(
        initial_capital=args.initial_capital,
        transaction_cost_rate=args.transaction_cost_rate,
        p_ratio=best_config["p_ratio"],
        q_stop_loss=best_config["q_stop_loss"],
        r_rising_ratio=best_config["r_rising_ratio"],
        risk_free_rate=args.risk_free_rate,
        rebalance_frequency=args.rebalance_frequency,
        prob_threshold=args.prob_threshold,
        variance_quantile=args.variance_quantile,
        variance_weight=args.variance_weight,
        market_variance_threshold=args.market_variance_threshold,
    )
    test_metrics = test_strategy.run_backtest(
        mean_rise_probabilities=mean_probs_by_split["eval"][:, :, 1],
        close_prices=close_prices_by_split["eval"],
        rise_variances=variances_by_split["eval"][:, :, 1],
    )
    print_metrics(eval_label, test_metrics)
    print(f"Avg filtered-out stocks/day: {test_metrics['avg_filtered_out']:.2f}")
    print(f"Skipped days due to market uncertainty: {int(test_metrics['skipped_days_due_to_market_uncertainty'])}")
    print(f"Average daily market variance: {test_metrics['avg_market_variance']:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = f"{args.model_version}_bayesian_{args.eval_split}"

    plot_path = output_dir / f"backtest_{mode_tag}.png"
    test_strategy.plot_results(
        title=f"{args.data_name} Backtest ({args.model_version}, bayesian)",
        save_path=str(plot_path),
    )

    variance_path = output_dir / f"variance_summary_{mode_tag}.pt"
    torch.save(
        {
            "val_variance": (
                None
                if "val" not in variances_by_split or variances_by_split["val"] is None
                else variances_by_split["val"].cpu()
            ),
            "evaluation_variance": variances_by_split["eval"].cpu(),
        },
        variance_path,
    )

    payload = {
        "data_name": args.data_name,
        "model_version": args.model_version,
        "prediction_mode": "mc",
        "num_mc_runs": args.num_mc_runs,
        "deterministic_val": args.deterministic_val,
        "rebalance_frequency": args.rebalance_frequency,
        "eval_split": args.eval_split,
        "prob_threshold": args.prob_threshold,
        "variance_quantile": args.variance_quantile,
        "variance_weight": args.variance_weight,
        "market_variance_threshold": args.market_variance_threshold,
        "best_validation_config": best_config,
        "validation_metrics": val_metrics,
        "evaluation_metrics": test_metrics,
        "variance_path": str(variance_path),
    }

    result_path = output_dir / f"backtest_{mode_tag}.json"
    result_path.write_text(json.dumps(make_json_safe(payload), indent=2))

    print(f"\nSaved plot to {plot_path}")
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
