from __future__ import annotations

import argparse
import importlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from Dataset import StockDataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_file(path: str, *, map_location: Optional[torch.device] = None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def split_data_by_time(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    total_date = data.shape[1]
    train_cutoff = int(total_date * 0.7)
    valid_cutoff = train_cutoff + int(total_date * 0.1)
    return {
        "train": data[:, :train_cutoff],
        "val": data[:, train_cutoff:valid_cutoff],
        "test": data[:, valid_cutoff:],
    }


def build_evaluation_split(split_data: Dict[str, torch.Tensor], eval_split: str) -> torch.Tensor:
    if eval_split == "test":
        return split_data["test"]
    if eval_split == "val_test":
        return torch.cat([split_data["val"], split_data["test"]], dim=1)
    raise ValueError(f"Unsupported eval_split: {eval_split}")


def normalize_split(split_data: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    split_mean = split_data.mean(dim=1, keepdim=True)
    split_std = split_data.std(dim=1, keepdim=True)
    return (split_data - split_mean) / (split_std + epsilon)


def generate_close_prices(split_data: torch.Tensor, lookback: int) -> torch.Tensor:
    return split_data[:, lookback - 1 :, 0]


def load_model(
    model_version: str,
    weight_path: str,
    data_shape: Tuple[int, int, int],
    dim: int,
    num_experts: int,
    num_heads_mha: int,
    num_channels: int,
    num_heads_causal_mha: int,
    lookback: int,
    num_mage: int,
    num_f2dattn: int,
    num_tch: int,
    topk: int,
    m1: int,
    num_s2dattn: int,
    num_gph: int,
    m2: int,
    dropout: float,
    device: torch.device,
) -> torch.nn.Module:
    model_module = importlib.import_module(model_version)
    model_cls = model_module.MaGNet
    n_stocks, _, n_features = data_shape
    model = model_cls(
        n_stocks,
        lookback,
        n_features,
        dim,
        num_mage,
        num_experts,
        num_heads_mha,
        num_f2dattn,
        num_channels,
        num_heads_causal_mha,
        num_tch,
        topk,
        m1,
        num_s2dattn,
        num_gph,
        m2,
        device=device,
        dropout=dropout,
    ).to(device)
    state_dict = load_file(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def predict_probabilities(
    model: torch.nn.Module,
    split_data: torch.Tensor,
    lookback: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    dataset = StockDataset(split_data, lookback, device)
    num_days = len(dataset)
    if num_days <= 0:
        raise ValueError("Split is too short for the requested lookback window.")

    sample_x, _ = dataset[0]
    n_stocks = sample_x.shape[0]
    model.eval()
    probabilities = torch.zeros(n_stocks, num_days, 2, device=device)
    with torch.no_grad():
        for day_idx in tqdm(range(num_days), desc="Deterministic inference"):
            x, _ = dataset[day_idx]
            logits, *_ = model(x)
            probabilities[:, day_idx, :] = torch.softmax(logits, dim=-1)
    return probabilities, None


class DailyPortfolioTradingStrategy:
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        transaction_cost_rate: float = 0.0025,
        p_ratio: float = 1.0,
        q_stop_loss: float = 0.5,
        r_rising_ratio: float = 1.0,
        risk_free_rate: float = 0.02,
        rebalance_frequency: int = 1,
    ):
        if not (0 < p_ratio <= 1):
            raise ValueError("p_ratio must satisfy 0 < p_ratio <= 1.")
        if not (0 < q_stop_loss < 1):
            raise ValueError("q_stop_loss must satisfy 0 < q_stop_loss < 1.")
        if not (0 <= r_rising_ratio <= 1):
            raise ValueError("r_rising_ratio must satisfy 0 <= r_rising_ratio <= 1.")
        if rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be at least 1.")

        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        self.p_ratio = p_ratio
        self.q_stop_loss = q_stop_loss
        self.r_rising_ratio = r_rising_ratio
        self.risk_free_rate = risk_free_rate
        self.rebalance_frequency = rebalance_frequency
        self.reset_tracking()

    def reset_tracking(self) -> None:
        self.portfolio_values: List[float] = []
        self.daily_returns: List[float] = []
        self.cumulative_returns: List[float] = []
        self.cash_history: List[float] = []
        self.transaction_costs: List[float] = []
        self.transaction_counts: List[int] = []
        self.daily_holdings: List[List[int]] = []

    def select_stocks(self, rise_probs: torch.Tensor) -> Tuple[List[int], int]:
        n_stocks = rise_probs.shape[0]
        target_stocks = max(1, int(self.p_ratio * n_stocks))
        rising_stocks = torch.where(rise_probs > 0.5)[0]
        m_rising = len(rising_stocks)
        stop_loss_threshold = int(target_stocks * self.q_stop_loss)

        if m_rising >= target_stocks:
            selected = torch.topk(rise_probs, target_stocks).indices.tolist()
            return selected, target_stocks

        if m_rising >= stop_loss_threshold:
            num_to_select = int(m_rising * self.r_rising_ratio)
            if num_to_select <= 0:
                return [], 0
            rising_probs = rise_probs[rising_stocks]
            top_indices = torch.topk(rising_probs, num_to_select).indices
            selected = rising_stocks[top_indices].tolist()
            return selected, num_to_select

        return [], 0

    def calculate_transaction_cost(self, trade_value: float) -> float:
        return trade_value * self.transaction_cost_rate

    def execute_trades(
        self,
        current_holdings: Dict[int, float],
        target_stocks: List[int],
        stock_prices: np.ndarray,
        cash: float,
    ) -> Tuple[Dict[int, float], float, float, int]:
        new_holdings: Dict[int, float] = {}
        total_cost = 0.0
        num_transactions = 0

        for stock_idx, shares in current_holdings.items():
            if stock_idx not in target_stocks:
                sale_value = shares * stock_prices[stock_idx]
                transaction_cost = self.calculate_transaction_cost(sale_value)
                cash += sale_value - transaction_cost
                total_cost += transaction_cost
                num_transactions += 1
            else:
                new_holdings[stock_idx] = shares

        portfolio_value = cash
        for stock_idx, shares in new_holdings.items():
            portfolio_value += shares * stock_prices[stock_idx]

        if target_stocks:
            target_value_per_stock = (
                portfolio_value / (1 + self.transaction_cost_rate) / len(target_stocks)
            )

            for stock_idx in target_stocks:
                current_value = new_holdings.get(stock_idx, 0.0) * stock_prices[stock_idx]
                target_value = target_value_per_stock

                if abs(current_value - target_value) <= 1e-6:
                    continue

                if current_value < target_value:
                    buy_value = target_value - current_value
                    shares_to_buy = buy_value / stock_prices[stock_idx]
                    transaction_cost = self.calculate_transaction_cost(buy_value)
                    new_holdings[stock_idx] = new_holdings.get(stock_idx, 0.0) + shares_to_buy
                    cash -= buy_value + transaction_cost
                    total_cost += transaction_cost
                    num_transactions += 1
                else:
                    sell_value = current_value - target_value
                    shares_to_sell = sell_value / stock_prices[stock_idx]
                    transaction_cost = self.calculate_transaction_cost(sell_value)
                    new_holdings[stock_idx] -= shares_to_sell
                    cash += sell_value - transaction_cost
                    total_cost += transaction_cost
                    num_transactions += 1

        return new_holdings, cash, total_cost, num_transactions

    def run_backtest(
        self,
        rise_probabilities: torch.Tensor,
        close_prices: torch.Tensor,
    ) -> Dict[str, float]:
        self.reset_tracking()

        rise_probs_np = (
            rise_probabilities.cpu().numpy()
            if isinstance(rise_probabilities, torch.Tensor)
            else rise_probabilities
        )
        close_prices_np = (
            close_prices.cpu().numpy() if isinstance(close_prices, torch.Tensor) else close_prices
        )

        n_stocks, t_days = rise_probs_np.shape
        if close_prices_np.shape[1] != t_days + 1:
            raise ValueError("close_prices must have one more day than rise probabilities.")

        cash = self.initial_capital
        holdings: Dict[int, float] = {}
        self.portfolio_values.append(self.initial_capital)
        self.cash_history.append(cash)

        for day_idx in range(t_days):
            day_rise_probs = torch.tensor(rise_probs_np[:, day_idx])
            current_prices = close_prices_np[:, day_idx]
            if day_idx % self.rebalance_frequency == 0:
                target_stocks, _ = self.select_stocks(day_rise_probs)
                holdings, cash, transaction_cost, num_trades = self.execute_trades(
                    holdings, target_stocks, current_prices, cash
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

        return self.calculate_performance_metrics()

    def calculate_performance_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["final_portfolio_value"] = self.portfolio_values[-1]
        metrics["cumulative_return"] = (
            self.portfolio_values[-1] / self.initial_capital - 1.0
            if self.portfolio_values
            else 0.0
        )

        trading_days = len(self.daily_returns)
        if trading_days > 0:
            returns_array = np.asarray(self.daily_returns, dtype=np.float64)
            metrics["annual_return"] = float(
                (self.portfolio_values[-1] / self.initial_capital) ** (252.0 / trading_days) - 1.0
            )
        else:
            returns_array = np.asarray([], dtype=np.float64)
            metrics["annual_return"] = 0.0

        if trading_days > 1:
            metrics["volatility"] = float(np.std(returns_array) * np.sqrt(252.0))
            if metrics["volatility"] > 0:
                metrics["sharpe_ratio"] = (
                    metrics["annual_return"] - self.risk_free_rate
                ) / metrics["volatility"]
            else:
                metrics["sharpe_ratio"] = 0.0

            peak = self.portfolio_values[0]
            max_drawdown = 0.0
            for value in self.portfolio_values[1:]:
                peak = max(peak, value)
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            metrics["max_drawdown"] = max_drawdown
            metrics["calmar_ratio"] = (
                metrics["annual_return"] / metrics["max_drawdown"]
                if metrics["max_drawdown"] > 0
                else float("inf")
            )
        else:
            metrics["volatility"] = 0.0
            metrics["sharpe_ratio"] = 0.0
            metrics["max_drawdown"] = 0.0
            metrics["calmar_ratio"] = 0.0

        metrics["avg_transaction_cost"] = (
            float(np.mean(self.transaction_costs)) if self.transaction_costs else 0.0
        )
        metrics["avg_num_trades"] = (
            float(np.mean(self.transaction_counts)) if self.transaction_counts else 0.0
        )
        return metrics

    def plot_results(self, title: str, save_path: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("matplotlib is not installed; skipping plot generation.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(title, fontsize=16)

        ax = axes[0]
        ax.plot(self.portfolio_values, linewidth=2, color="blue")
        ax.set_title("Portfolio Value")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.3)
        ax.axhline(
            y=self.initial_capital,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Initial Capital",
        )
        ax.legend()

        ax = axes[1]
        if self.daily_returns:
            daily_returns_pct = [100.0 * daily_return for daily_return in self.daily_returns]
            ax.hist(daily_returns_pct, bins=50, alpha=0.7, color="slateblue", edgecolor="black")
            ax.set_title("Daily Returns Distribution")
            ax.set_xlabel("Daily Return (%)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            mean_return = float(np.mean(daily_returns_pct))
            ax.axvline(x=mean_return, color="red", linestyle="--", label=f"Mean: {mean_return:.2f}%")
            ax.legend()

        ax = axes[2]
        drawdowns = []
        peak = self.portfolio_values[0]
        for value in self.portfolio_values:
            peak = max(peak, value)
            drawdowns.append((value - peak) / peak * 100.0)
        ax.fill_between(range(len(drawdowns)), drawdowns, 0.0, color="red", alpha=0.3)
        ax.plot(drawdowns, color="darkred", linewidth=1)
        ax.set_title("Drawdown")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0.0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def grid_search_strategy(
    rise_probabilities: torch.Tensor,
    close_prices: torch.Tensor,
    p_grid: Sequence[float],
    q_grid: Sequence[float],
    r_grid: Sequence[float],
    initial_capital: float,
    transaction_cost_rate: float,
    risk_free_rate: float,
    rebalance_frequency: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    best_config: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_objective = (-math.inf, -math.inf)

    all_configs = list(product(p_grid, q_grid, r_grid))
    for p_ratio, q_stop_loss, r_rising_ratio in tqdm(all_configs, desc="Grid search"):
        strategy = DailyPortfolioTradingStrategy(
            initial_capital=initial_capital,
            transaction_cost_rate=transaction_cost_rate,
            p_ratio=p_ratio,
            q_stop_loss=q_stop_loss,
            r_rising_ratio=r_rising_ratio,
            risk_free_rate=risk_free_rate,
            rebalance_frequency=rebalance_frequency,
        )
        metrics = strategy.run_backtest(rise_probabilities, close_prices)
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
        raise RuntimeError("Grid search did not evaluate any configurations.")

    return best_config, best_metrics


def parse_grid(raw_values: Sequence[float], fallback_start: float, fallback_end: float, step: float) -> List[float]:
    if raw_values:
        return [float(value) for value in raw_values]

    values: List[float] = []
    current = fallback_start
    while current <= fallback_end + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def print_metrics(header: str, metrics: Dict[str, float]) -> None:
    print(f"\n{header}")
    print("-" * len(header))
    print(f"Final Portfolio Value: {metrics['final_portfolio_value']:,.2f}")
    print(f"Cumulative Return:     {metrics['cumulative_return'] * 100:.2f}%")
    print(f"Annual Return:         {metrics['annual_return'] * 100:.2f}%")
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.4f}")
    print(f"Calmar Ratio:          {metrics['calmar_ratio']:.4f}")
    print(f"Max Drawdown:          {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Annualized Volatility: {metrics['volatility'] * 100:.2f}%")
    print(f"Avg Transaction Cost:  {metrics['avg_transaction_cost']:.2f}")
    print(f"Avg # Trades / Day:    {metrics['avg_num_trades']:.2f}")


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic baseline backtests for MaGNet variants.")
    parser.add_argument("--model-version", default="Magnetv1")
    parser.add_argument("--data-name", default="NASDAQ100")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--weight-path", required=True)
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

    probabilities_by_split: Dict[str, torch.Tensor] = {}
    variances_by_split: Dict[str, Optional[torch.Tensor]] = {}
    close_prices_by_split: Dict[str, torch.Tensor] = {}

    eval_label = "Test Metrics" if args.eval_split == "test" else "Combined Val+Test Metrics"
    eval_raw_split = build_evaluation_split(split_data, args.eval_split)
    print(f"\nPreparing {args.eval_split} evaluation split...")
    eval_normalized_split = normalize_split(eval_raw_split)
    eval_probabilities, eval_variances = predict_probabilities(
        model=model,
        split_data=eval_normalized_split,
        lookback=args.lookback,
        device=device,
    )
    probabilities_by_split["eval"] = eval_probabilities
    variances_by_split["eval"] = eval_variances
    close_prices_by_split["eval"] = generate_close_prices(eval_raw_split, args.lookback)

    fixed_config = (
        args.fixed_p_ratio is not None
        and args.fixed_q_stop_loss is not None
        and args.fixed_r_rising_ratio is not None
    )

    needs_validation = (not fixed_config) or args.eval_split == "test"

    if needs_validation:
        print("\nPreparing val split...")
        raw_split = split_data["val"]
        normalized_split = normalize_split(raw_split)
        probabilities, variances = predict_probabilities(
            model=model,
            split_data=normalized_split,
            lookback=args.lookback,
            device=device,
        )
        probabilities_by_split["val"] = probabilities
        variances_by_split["val"] = variances
        close_prices_by_split["val"] = generate_close_prices(raw_split, args.lookback)

    if fixed_config:
        best_config = {
            "p_ratio": args.fixed_p_ratio,
            "q_stop_loss": args.fixed_q_stop_loss,
            "r_rising_ratio": args.fixed_r_rising_ratio,
        }
        print("\nUsing fixed config:", best_config)
        val_metrics = None
        if args.eval_split == "test":
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
                probabilities_by_split["val"][:, :, 1],
                close_prices_by_split["val"],
            )
    else:
        p_grid = parse_grid(args.p_grid, 0.05, 1.0, 0.05)
        q_grid = parse_grid(args.q_grid, 0.05, 0.95, 0.05)
        r_grid = parse_grid(args.r_grid, 0.0, 1.0, 0.05)

        best_config, val_metrics = grid_search_strategy(
            rise_probabilities=probabilities_by_split["val"][:, :, 1],
            close_prices=close_prices_by_split["val"],
            p_grid=p_grid,
            q_grid=q_grid,
            r_grid=r_grid,
            initial_capital=args.initial_capital,
            transaction_cost_rate=args.transaction_cost_rate,
            risk_free_rate=args.risk_free_rate,
            rebalance_frequency=args.rebalance_frequency,
        )

        print("\nBest validation config:", best_config)

    if val_metrics is not None:
        print_metrics("Validation Metrics", val_metrics)

    test_strategy = DailyPortfolioTradingStrategy(
        initial_capital=args.initial_capital,
        transaction_cost_rate=args.transaction_cost_rate,
        p_ratio=best_config["p_ratio"],
        q_stop_loss=best_config["q_stop_loss"],
        r_rising_ratio=best_config["r_rising_ratio"],
        risk_free_rate=args.risk_free_rate,
        rebalance_frequency=args.rebalance_frequency,
    )
    test_metrics = test_strategy.run_backtest(
        probabilities_by_split["eval"][:, :, 1],
        close_prices_by_split["eval"],
    )
    print_metrics(eval_label, test_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = f"{args.model_version}_baseline_{args.eval_split}"

    plot_path = output_dir / f"backtest_{mode_tag}.png"
    test_strategy.plot_results(
        title=f"{args.data_name} Backtest ({args.model_version}, baseline)",
        save_path=str(plot_path),
    )

    payload = {
        "data_name": args.data_name,
        "model_version": args.model_version,
        "prediction_mode": "deterministic",
        "rebalance_frequency": args.rebalance_frequency,
        "eval_split": args.eval_split,
        "best_validation_config": best_config,
        "validation_metrics": val_metrics,
        "evaluation_metrics": test_metrics,
    }

    result_path = output_dir / f"backtest_{mode_tag}.json"
    result_path.write_text(json.dumps(make_json_safe(payload), indent=2))

    print(f"\nSaved plot to {plot_path}")
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
