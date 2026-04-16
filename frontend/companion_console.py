from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.aidssist_runtime.solver_orchestrator import submit_solve_run
from backend.workflow_store import WorkflowStore


def _run_table(runs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run.run_id,
                "route": run.route,
                "status": run.status,
                "query": run.query,
                "queued_at": run.queued_at,
                "finished_at": run.finished_at,
                "elapsed_ms": run.elapsed_ms,
            }
            for run in runs
        ]
    )


def render_companion_console(*, workflow_store: WorkflowStore, api_ready: bool, api_message: str) -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-shell__copy">
                <span class="section-kicker">Companion Console</span>
                <h2 class="hero-title">Power-user diagnostics for the Solver Platform</h2>
                <p class="hero-caption">Inspect workspaces, traces, validator reports, retrieval memory, and manual replay controls without leaving Streamlit.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if api_ready:
        st.success(api_message)
    else:
        st.warning(api_message)

    workspaces = workflow_store.list_workspaces(limit=100)
    if not workspaces:
        st.info("No solver workspaces have been created yet. Use the React app to create one and upload assets.")
        return

    workspace_lookup = {f"{workspace.name} ({workspace.workspace_id[:8]})": workspace for workspace in workspaces}
    selected_workspace_label = st.selectbox(
        "Workspace",
        options=list(workspace_lookup.keys()),
        key="companion_workspace",
    )
    workspace = workspace_lookup[selected_workspace_label]
    assets = workflow_store.list_workspace_assets(workspace.workspace_id, limit=100)
    runs = workflow_store.list_workspace_solve_runs(workspace.workspace_id, limit=30)
    derived_datasets = workflow_store.list_workspace_derived_datasets(workspace.workspace_id, limit=30)
    chunks = workflow_store.list_workspace_chunks(workspace.workspace_id, limit=60)
    embeddings = workflow_store.list_embeddings_for_chunk_ids([chunk.chunk_id for chunk in chunks])

    metrics = st.columns(5)
    metrics[0].metric("Assets", f"{len(assets):,}")
    metrics[1].metric("Derived datasets", f"{len(derived_datasets):,}")
    metrics[2].metric("Solve runs", f"{len(runs):,}")
    metrics[3].metric("Chunks", f"{len(chunks):,}")
    metrics[4].metric("Embeddings", f"{len(embeddings):,}")

    monitor_tab, trace_tab, chunk_tab, replay_tab = st.tabs(
        ["Job Monitor", "Trace Inspector", "Chunk Memory", "Manual Replay"]
    )

    with monitor_tab:
        with st.container(border=True):
            st.markdown("### Recent solve runs")
            if runs:
                st.dataframe(_run_table(runs), use_container_width=True, hide_index=True)
            else:
                st.info("No solve runs have been recorded for this workspace yet.")

    with trace_tab:
        if not runs:
            st.info("Run a solve job first to inspect its trace.")
        else:
            run_lookup = {f"{run.status.upper()} · {run.query[:48]}": run for run in runs}
            selected_run = run_lookup[
                st.selectbox("Solve run", options=list(run_lookup.keys()), key="companion_run")
            ]
            steps = workflow_store.list_solve_steps(selected_run.run_id)
            reports = workflow_store.list_validator_reports(selected_run.run_id)
            st.markdown("### Plan")
            st.code(selected_run.plan_text or "No plan text was persisted for this run.", language="text")
            trace_columns = st.columns(2, gap="large")
            with trace_columns[0]:
                with st.container(border=True):
                    st.markdown("**Retrieval trace**")
                    st.json(selected_run.retrieval_trace)
            with trace_columns[1]:
                with st.container(border=True):
                    st.markdown("**Final output**")
                    st.json(selected_run.final_output or {})
            if steps:
                with st.container(border=True):
                    st.markdown("**Recorded steps**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "index": step.step_index,
                                    "stage": step.stage,
                                    "status": step.status,
                                    "title": step.title,
                                }
                                for step in steps
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            if reports:
                with st.container(border=True):
                    st.markdown("**Validator diagnostics**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "attempt": report.attempt_index + 1,
                                    "status": report.status,
                                    "error_message": report.error_message,
                                    "checks": ", ".join(check["name"] for check in report.checks),
                                }
                                for report in reports
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    with chunk_tab:
        with st.container(border=True):
            st.markdown("### Retrieval memory")
            if chunks:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "chunk_id": chunk.chunk_id,
                                "title": chunk.title,
                                "dataset_id": chunk.dataset_id,
                                "tokens": chunk.token_count,
                                "file_hint": chunk.metadata.get("file_name"),
                            }
                            for chunk in chunks
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No chunks are stored yet for this workspace.")

    with replay_tab:
        with st.container(border=True):
            st.markdown("### Manual replay")
            asset_lookup = {"Workspace-wide context": None}
            asset_lookup.update({asset.title: asset.asset_id for asset in assets})
            selected_asset_label = st.selectbox(
                "Anchor asset",
                options=list(asset_lookup.keys()),
                key="companion_replay_asset",
            )
            replay_query = st.text_area(
                "Replay query",
                key="companion_replay_query",
                placeholder="Example: Redesign this uploaded project and explain the schema changes.",
                height=140,
            )
            if st.button("Run manual solve replay", type="primary", use_container_width=True):
                if not replay_query.strip():
                    st.warning("Enter a replay query first.")
                else:
                    try:
                        run = submit_solve_run(
                            workspace_id=workspace.workspace_id,
                            query=replay_query.strip(),
                            user_id=workspace.user_id,
                            asset_id=asset_lookup[selected_asset_label],
                        )
                        st.success(f"Manual replay submitted as run {run.run_id}.")
                    except Exception as error:
                        st.error(str(error))
