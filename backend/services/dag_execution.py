from __future__ import annotations

from typing import Any


def _normalize_dependencies(value: Any, *, valid_steps: set[int]) -> list[int]:
    dependencies: list[int] = []
    for item in list(value or []):
        try:
            dependency = int(item)
        except (TypeError, ValueError):
            continue
        if dependency > 0 and dependency in valid_steps and dependency not in dependencies:
            dependencies.append(dependency)
    return sorted(dependencies)


def build_execution_graph(plan: list[dict[str, Any]] | None) -> dict[str, Any]:
    normalized_steps = [dict(step) for step in list(plan or []) if isinstance(step, dict)]
    if not normalized_steps:
        return {
            "nodes": {},
            "dependencies": {},
            "dependents": {},
            "batches": [],
            "terminal_nodes": [],
            "has_cycle": False,
        }

    step_ids = {
        max(1, int(step.get("step") or index))
        for index, step in enumerate(normalized_steps, start=1)
    }
    nodes: dict[int, dict[str, Any]] = {}
    dependencies: dict[int, list[int]] = {}
    dependents: dict[int, list[int]] = {step_id: [] for step_id in step_ids}

    for index, step in enumerate(normalized_steps, start=1):
        step_id = max(1, int(step.get("step") or index))
        node = dict(step)
        node["step"] = step_id
        node["depends_on"] = _normalize_dependencies(node.get("depends_on"), valid_steps=step_ids)
        nodes[step_id] = node
        dependencies[step_id] = list(node["depends_on"])

    for step_id, step_dependencies in dependencies.items():
        for dependency in step_dependencies:
            dependents.setdefault(dependency, []).append(step_id)

    remaining_dependencies = {
        step_id: set(step_dependencies)
        for step_id, step_dependencies in dependencies.items()
    }
    ready_nodes = sorted(step_id for step_id, step_dependencies in remaining_dependencies.items() if not step_dependencies)
    seen_nodes: set[int] = set()
    batches: list[list[int]] = []

    while ready_nodes:
        current_batch = sorted(step_id for step_id in ready_nodes if step_id not in seen_nodes)
        if not current_batch:
            break
        batches.append(current_batch)
        next_ready: set[int] = set()
        for step_id in current_batch:
            seen_nodes.add(step_id)
            for dependent in dependents.get(step_id, []):
                remaining_dependencies.setdefault(dependent, set()).discard(step_id)
                if not remaining_dependencies.get(dependent) and dependent not in seen_nodes:
                    next_ready.add(dependent)
        ready_nodes = sorted(next_ready)

    terminal_nodes = sorted(
        step_id
        for step_id in nodes
        if not dependents.get(step_id)
    )
    has_cycle = len(seen_nodes) != len(nodes)

    return {
        "nodes": nodes,
        "dependencies": dependencies,
        "dependents": dependents,
        "batches": batches,
        "terminal_nodes": terminal_nodes,
        "has_cycle": has_cycle,
    }


def find_parallel_batches(plan: list[dict[str, Any]] | None) -> list[list[int]]:
    graph = build_execution_graph(plan)
    return [list(batch) for batch in graph.get("batches", []) if len(batch) > 1]
