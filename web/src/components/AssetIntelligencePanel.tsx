import { useEffect, useState } from "react";
import { Bot, GitBranch, Network, RefreshCw, Sparkles, Table2 } from "lucide-react";

import { generateAIInsights, getAssetIntelligence } from "../lib/api";
import type { AssetIntelligence, InsightChart } from "../types/api";

type AssetIntelligencePanelProps = {
  assetId: string | null;
  token: string | null;
  onAskQuestion?: (question: string) => void;
};

function formatCount(value: number | undefined): string {
  return new Intl.NumberFormat().format(value || 0);
}

function toNumeric(value: unknown): number {
  return typeof value === "number" ? value : Number(value || 0);
}

function MiniInsightChart({ chart }: { chart: InsightChart | null | undefined }) {
  if (!chart || !chart.rows.length) {
    return null;
  }

  if (chart.type === "line") {
    const values = chart.rows.map((row) => toNumeric(row[chart.y || "value"]));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const points = values
      .map((value, index) => {
        const x = (index / Math.max(values.length - 1, 1)) * 100;
        const y = max === min ? 50 : 100 - ((value - min) / (max - min)) * 100;
        return `${x},${y}`;
      })
      .join(" ");
    return (
      <div className="mini-line-chart">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none">
          <polyline points={points} />
        </svg>
      </div>
    );
  }

  const max = Math.max(...chart.rows.map((row) => toNumeric(row[chart.y || "value"])), 1);
  return (
    <div className="mini-bar-stack">
      {chart.rows.slice(0, 5).map((row, index) => (
        <div key={`${chart.title || chart.type}-${index}`} className="mini-bar-row">
          <span>{String(row[chart.x] ?? "")}</span>
          <div className="mini-bar-row__track">
            <div
              className="mini-bar-row__fill"
              style={{ width: `${(toNumeric(row[chart.y || "value"]) / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function InsightBlock({
  title,
  items
}: {
  title: string;
  items: Array<{ title: string; narrative: string; confidence?: string | null; chart?: InsightChart | null }>;
}) {
  if (!items.length) {
    return null;
  }

  return (
    <section className="intelligence-section">
      <div className="intelligence-section__header">
        <span className="section-kicker">{title}</span>
      </div>
      <div className="insight-card-grid">
        {items.map((item, index) => (
          <article key={`${item.title}-${index}`} className="insight-premium-card">
            <div className="insight-premium-card__topline">
              <Sparkles size={15} />
              <span>{item.confidence || "curated insight"}</span>
            </div>
            <h4>{item.title}</h4>
            <p>{item.narrative}</p>
            <MiniInsightChart chart={item.chart} />
          </article>
        ))}
      </div>
    </section>
  );
}

export default function AssetIntelligencePanel({
  assetId,
  token,
  onAskQuestion
}: AssetIntelligencePanelProps) {
  const [intelligence, setIntelligence] = useState<AssetIntelligence | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!assetId || !token) {
      setIntelligence(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void (async () => {
      try {
        const payload = await getAssetIntelligence(assetId, token);
        if (!cancelled) {
          setIntelligence(payload);
        }
      } catch (caughtError) {
        if (!cancelled) {
          setError(caughtError instanceof Error ? caughtError.message : "Asset intelligence could not be loaded.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [assetId, token]);

  async function handleRefresh() {
    if (!assetId || !token) {
      return;
    }
    setRefreshing(true);
    setError(null);
    try {
      const payload = await generateAIInsights({ asset_id: assetId, force_refresh: true }, token);
      setIntelligence(payload);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "AI insights could not be refreshed.");
    } finally {
      setRefreshing(false);
    }
  }

  if (!assetId) {
    return (
      <article className="surface-card surface-card--wide">
        <span className="section-kicker">AI Insights</span>
        <h3>Select an asset to unlock schema mapping and AI insight generation</h3>
        <p>As soon as a dataset asset is selected, Aidssist will profile tables, map relationships, and surface insight cards.</p>
      </article>
    );
  }

  if (loading) {
    return (
      <article className="surface-card surface-card--wide">
        <span className="section-kicker">AI Insights</span>
        <h3>Preparing the intelligence layer...</h3>
        <p>Building the DuckDB session, schema graph, and AI insight cards for this asset.</p>
      </article>
    );
  }

  if (error) {
    return (
      <article className="surface-card surface-card--wide">
        <span className="section-kicker">AI Insights</span>
        <h3>Asset intelligence is unavailable</h3>
        <div className="notice-card notice-card--error">{error}</div>
      </article>
    );
  }

  if (!intelligence) {
    return null;
  }

  return (
    <article className="surface-card surface-card--wide intelligence-shell">
      <div className="intelligence-shell__header">
        <div>
          <span className="section-kicker">AI Insights</span>
          <h3>{(intelligence.dataset_type || intelligence.schema.dataset_type || "dataset").replace(/_/g, " ")} intelligence</h3>
          <p>{intelligence.insights.summary || "Aidssist has profiled the selected asset and generated a first-pass intelligence layer."}</p>
        </div>
        <button type="button" className="ghost-button" onClick={() => void handleRefresh()} disabled={refreshing}>
          <RefreshCw size={15} className={refreshing ? "spin" : ""} />
          {refreshing ? "Refreshing..." : "Refresh AI"}
        </button>
      </div>

      <div className="intelligence-metric-grid">
        <div className="intelligence-metric-card">
          <Table2 size={18} />
          <span>Tables</span>
          <strong>{formatCount(intelligence.schema.tables.length)}</strong>
        </div>
        <div className="intelligence-metric-card">
          <Network size={18} />
          <span>Relationships</span>
          <strong>{formatCount(intelligence.schema.relationships.length)}</strong>
        </div>
        <div className="intelligence-metric-card">
          <Sparkles size={18} />
          <span>Insights</span>
          <strong>{formatCount(intelligence.insights.insights.length + intelligence.insights.anomalies.length)}</strong>
        </div>
        <div className="intelligence-metric-card">
          <Bot size={18} />
          <span>Ask-data prompts</span>
          <strong>{formatCount(intelligence.chat_context.suggested_questions?.length)}</strong>
        </div>
      </div>

      <section className="intelligence-section">
        <div className="intelligence-section__header">
          <span className="section-kicker">Dataset Graph</span>
        </div>
        <div className="dataset-graph-surface">
          <div className="dataset-graph-surface__nodes">
            {intelligence.schema.graph.nodes.map((node) => (
              <article key={node.id} className="graph-node-card">
                <strong>{node.label}</strong>
                <span>{formatCount(node.row_count)} rows</span>
                <small>{formatCount(node.column_count)} columns</small>
              </article>
            ))}
          </div>
          <div className="dataset-graph-surface__edges">
            {intelligence.schema.relationships.length ? (
              intelligence.schema.relationships.map((edge) => (
                <div
                  key={`${edge.left_table}-${edge.left_column}-${edge.right_table}-${edge.right_column}`}
                  className="graph-edge-pill"
                >
                  <GitBranch size={14} />
                  <span>{`${edge.left_table}.${edge.left_column} -> ${edge.right_table}.${edge.right_column}`}</span>
                  <small>{Math.round(edge.confidence * 100)}%</small>
                </div>
              ))
            ) : (
              <div className="inline-note">Aidssist did not detect strong foreign-key relationships yet.</div>
            )}
          </div>
        </div>
      </section>

      <InsightBlock title="Key Insights" items={intelligence.insights.insights} />
      <InsightBlock title="Anomalies" items={intelligence.insights.anomalies} />

      {intelligence.insights.recommendations.length ? (
        <section className="intelligence-section">
          <div className="intelligence-section__header">
            <span className="section-kicker">Recommendations</span>
          </div>
          <div className="recommendation-list">
            {intelligence.insights.recommendations.map((item, index) => (
              <article key={`${item.title}-${index}`} className={`recommendation-card ${index === 0 ? "recommendation-card--best" : ""}`}>
                <strong>{item.title}</strong>
                <p>{item.body}</p>
                {item.priority ? <small>{item.priority} priority</small> : null}
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {intelligence.chat_context.suggested_questions?.length ? (
        <section className="intelligence-section">
          <div className="intelligence-section__header">
            <span className="section-kicker">Suggested Questions</span>
          </div>
          <div className="question-chip-row">
            {intelligence.chat_context.suggested_questions.map((question) => (
              <button
                key={question}
                type="button"
                className="question-chip"
                onClick={() => onAskQuestion?.(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </section>
      ) : null}
    </article>
  );
}
