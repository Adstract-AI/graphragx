"""One-time repair for aggregate W&B panels rendered as history lines.

Older GraphRAGX runs logged aggregate ``Summary_Plots/*`` and
``Run_Summary/*`` values through run history. W&B therefore generated line
plots with ``Step`` on the x-axis. The values are also present in each run's
summary, so the correct repair is to make those workspace panels read summary
scalars and render them as bars. Historical rows are intentionally left
untouched.

This script creates a new, filtered W&B saved view for explicitly selected
runs. It never overwrites the source workspace and never changes run data. It
is a dry run unless ``--apply`` is supplied. ``wandb-workspaces`` is an
optional maintenance dependency and is deliberately not part of the project
runtime dependencies.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_PREFIXES = ("Summary_Plots/", "Run_Summary/")


@dataclass(frozen=True)
class PanelRepair:
    """Description of one workspace panel mutation."""

    section: str
    title: str
    metrics: tuple[str, ...]
    action: str


def _load_workspace_modules() -> tuple[Any, Any, Any]:
    try:
        workspaces = importlib.import_module("wandb_workspaces.workspaces")
        reports = importlib.import_module("wandb_workspaces.reports.v2")
        wandb = importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError(
            "This one-time script requires wandb-workspaces. Run it with "
            "`uv run --with wandb-workspaces python "
            "scripts/fix_wandb_summary_metric_panels.py ...`."
        ) from error
    return workspaces, reports, wandb


def resolve_selected_runs(
    *,
    wandb: Any,
    entity: str,
    project: str,
    run_ids: list[str] | None,
    run_names: list[str] | None,
) -> list[tuple[str, str]]:
    """Resolve and validate the exact W&B runs included in the repaired view."""
    api = wandb.Api()
    if run_ids:
        selected: list[tuple[str, str]] = []
        for run_id in dict.fromkeys(run_ids):
            run = api.run(f"{entity}/{project}/{run_id}")
            selected.append((str(run.id), str(run.name)))
        return selected

    requested_names = list(dict.fromkeys(run_names or []))
    requested = set(requested_names)
    matches: dict[str, list[Any]] = {name: [] for name in requested_names}
    for run in api.runs(f"{entity}/{project}"):
        if run.name in requested:
            matches[run.name].append(run)
    missing = [name for name, runs in matches.items() if not runs]
    duplicates = [name for name, runs in matches.items() if len(runs) > 1]
    if missing:
        raise ValueError(f"W&B run names not found: {', '.join(missing)}")
    if duplicates:
        raise ValueError(
            "W&B run names are not unique; select these by --wandb-run-id: "
            + ", ".join(duplicates)
        )
    return [(str(matches[name][0].id), name) for name in requested_names]


def _metric_name(metric: object) -> str | None:
    if isinstance(metric, str):
        return metric
    name = getattr(metric, "name", None)
    return name if isinstance(name, str) else None


def _is_target_metric(
    metric_name: str,
    *,
    prefixes: tuple[str, ...],
    exact_metrics: frozenset[str],
) -> bool:
    return metric_name in exact_metrics or metric_name.startswith(prefixes)


def _line_to_summary_bar(panel: Any, *, reports: Any, orientation: str) -> Any:
    metric_names = [_metric_name(metric) for metric in panel.y]
    if any(name is None for name in metric_names):
        raise ValueError("Line panel contains an unsupported metric selector.")
    return reports.BarPlot(
        title=panel.title,
        metrics=[reports.SummaryMetric(name) for name in metric_names],
        orientation=orientation,
        range_x=panel.range_y,
        title_x=panel.title_y,
        groupby=panel.groupby,
        groupby_aggfunc=panel.groupby_aggfunc,
        groupby_rangefunc=panel.groupby_rangefunc,
        max_runs_to_show=panel.max_runs_to_show,
        custom_expressions=panel.custom_expressions,
        legend_template=panel.legend_template,
        font_size=panel.font_size,
        line_titles=panel.line_titles,
        line_colors=panel.line_colors,
        aggregate=panel.aggregate,
        layout=copy.deepcopy(panel.layout),
    )


def _bar_to_explicit_summary(panel: Any, *, reports: Any) -> Any:
    metric_names = [_metric_name(metric) for metric in panel.metrics]
    if any(name is None for name in metric_names):
        raise ValueError("Bar panel contains an unsupported metric selector.")
    return reports.BarPlot(
        title=panel.title,
        metrics=[reports.SummaryMetric(name) for name in metric_names],
        orientation=panel.orientation,
        range_x=panel.range_x,
        title_x=panel.title_x,
        title_y=panel.title_y,
        groupby=panel.groupby,
        groupby_aggfunc=panel.groupby_aggfunc,
        groupby_rangefunc=panel.groupby_rangefunc,
        max_runs_to_show=panel.max_runs_to_show,
        max_bars_to_show=panel.max_bars_to_show,
        custom_expressions=panel.custom_expressions,
        legend_template=panel.legend_template,
        font_size=panel.font_size,
        line_titles=panel.line_titles,
        line_colors=panel.line_colors,
        aggregate=panel.aggregate,
        layout=copy.deepcopy(panel.layout),
    )


def repair_workspace_panels(
    workspace: Any,
    *,
    reports: Any,
    prefixes: tuple[str, ...],
    exact_metrics: frozenset[str],
    orientation: str,
) -> list[PanelRepair]:
    """Convert matching line panels and explicitly bind bars to summaries."""
    repairs: list[PanelRepair] = []
    for section in workspace.sections:
        repaired_panels: list[Any] = []
        for panel in section.panels:
            if isinstance(panel, reports.LinePlot):
                metric_names = tuple(
                    name
                    for metric in panel.y
                    if (name := _metric_name(metric)) is not None
                )
                if metric_names and all(
                    _is_target_metric(
                        name,
                        prefixes=prefixes,
                        exact_metrics=exact_metrics,
                    )
                    for name in metric_names
                ):
                    repaired_panels.append(
                        _line_to_summary_bar(
                            panel,
                            reports=reports,
                            orientation=orientation,
                        )
                    )
                    repairs.append(
                        PanelRepair(
                            section=section.name,
                            title=panel.title or metric_names[0],
                            metrics=metric_names,
                            action="line->summary-bar",
                        )
                    )
                    continue
            elif isinstance(panel, reports.BarPlot):
                metric_names = tuple(
                    name
                    for metric in panel.metrics
                    if (name := _metric_name(metric)) is not None
                )
                if metric_names and all(
                    _is_target_metric(
                        name,
                        prefixes=prefixes,
                        exact_metrics=exact_metrics,
                    )
                    for name in metric_names
                ) and not all(
                    isinstance(metric, reports.SummaryMetric)
                    for metric in panel.metrics
                ):
                    repaired_panels.append(
                        _bar_to_explicit_summary(panel, reports=reports)
                    )
                    repairs.append(
                        PanelRepair(
                            section=section.name,
                            title=panel.title or metric_names[0],
                            metrics=metric_names,
                            action="bar->summary-bar",
                        )
                    )
                    continue
            repaired_panels.append(panel)
        section.panels = repaired_panels
    return repairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-time conversion of aggregate W&B history panels into "
            "summary-scalar bar panels. Without --apply, the workspace is not saved."
        )
    )
    parser.add_argument(
        "--workspace-url",
        required=True,
        help=(
            "Full W&B workspace or saved-view URL, including its `nw` query "
            "parameter."
        ),
    )
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument(
        "--wandb-run-id",
        action="append",
        help="Exact W&B run ID to include; repeat for every run to repair.",
    )
    selectors.add_argument(
        "--wandb-run-name",
        action="append",
        help=(
            "Exact W&B run name to include; repeat for every run. Names must be "
            "unique in the project."
        ),
    )
    parser.add_argument(
        "--view-name",
        default="GraphRAGX repaired summary metrics",
        help="Name of the new filtered saved view.",
    )
    parser.add_argument(
        "--metric-prefix",
        action="append",
        default=None,
        help=(
            "Metric prefix to repair; repeat as needed. Defaults to "
            "Summary_Plots/ and Run_Summary/."
        ),
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Exact metric key to repair in addition to selected prefixes.",
    )
    parser.add_argument(
        "--orientation",
        choices=("h", "v"),
        default="h",
        help="Bar orientation for converted line panels (default: horizontal).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create the filtered repaired view. Omit for a read-only dry run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workspaces, reports, wandb = _load_workspace_modules()
        workspace = workspaces.Workspace.from_url(args.workspace_url)
        selected_runs = resolve_selected_runs(
            wandb=wandb,
            entity=workspace.entity,
            project=workspace.project,
            run_ids=args.wandb_run_id,
            run_names=args.wandb_run_name,
        )
        selected_ids = [run_id for run_id, _ in selected_runs]
        workspace.name = args.view_name
        workspace.runset_settings.filters = (
            f'Metric("ID") in {selected_ids!r}'
        )
        workspace.runset_settings.pinned_runs = selected_ids[:20]
        for run_id in selected_ids:
            existing = workspace.runset_settings.run_settings.get(run_id)
            if existing is None:
                workspace.runset_settings.run_settings[run_id] = (
                    workspaces.RunSettings(disabled=False)
                )
            else:
                existing.disabled = False
        prefixes = tuple(args.metric_prefix or DEFAULT_PREFIXES)
        exact_metrics = frozenset(args.metric or ())
        repairs = repair_workspace_panels(
            workspace,
            reports=reports,
            prefixes=prefixes,
            exact_metrics=exact_metrics,
            orientation=args.orientation,
        )
    except Exception as error:
        print(f"Could not inspect W&B workspace: {error}", file=sys.stderr)
        return 2

    if not repairs:
        print("No matching W&B panels require repair.")
        return 0

    print("Selected runs (and no others):")
    for run_id, run_name in selected_runs:
        print(f"  {run_id}  {run_name}")

    for repair in repairs:
        print(
            f"{'APPLY' if args.apply else 'DRY RUN'} "
            f"section={repair.section!r} title={repair.title!r} "
            f"action={repair.action} metrics={','.join(repair.metrics)}"
        )
    print(f"Matching panels: {len(repairs)}")

    if not args.apply:
        print("Dry run only; rerun with --apply to create the new saved view.")
        return 0
    try:
        workspace.save_as_new_view()
    except Exception as error:
        print(f"Could not save W&B workspace: {error}", file=sys.stderr)
        return 1
    print(f"CREATED filtered repaired view: {workspace.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
