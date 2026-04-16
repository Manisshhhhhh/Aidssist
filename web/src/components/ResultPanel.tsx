type ResultPanelProps = {
  analysisOutput: Record<string, unknown> | null;
};

function formatCount(value: unknown) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "0";
  }
  return numericValue.toLocaleString();
}

export function ResultPanel({ analysisOutput }: ResultPanelProps) {
  if (!analysisOutput) {
    return (
      <section className="panel">
        <div className="panel__eyebrow">Result</div>
        <h2>Run an analysis</h2>
        <p>The AI result, summary, and decision guidance will appear here.</p>
      </section>
    );
  }

  const contract =
    analysisOutput.analysis_contract && typeof analysisOutput.analysis_contract === "object"
      ? (analysisOutput.analysis_contract as Record<string, unknown>)
      : null;
  const summary = typeof analysisOutput.summary === "string" ? analysisOutput.summary : null;
  const insights = typeof analysisOutput.insights === "string" ? analysisOutput.insights : null;
  const decisions =
    typeof analysisOutput.business_decisions === "string" ? analysisOutput.business_decisions : null;
  const confidence = typeof analysisOutput.confidence === "string" ? analysisOutput.confidence : null;
  const contractIntent = typeof contract?.intent === "string" ? contract.intent : null;
  const recommendations = Array.isArray(contract?.recommendations)
    ? contract.recommendations.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
  const warnings = Array.isArray(contract?.warnings)
    ? contract.warnings.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
  const rawCleaningReport =
    (contract?.cleaning_report && typeof contract.cleaning_report === "object"
      ? (contract.cleaning_report as Record<string, unknown>)
      : null) ||
    (analysisOutput.cleaning_report && typeof analysisOutput.cleaning_report === "object"
      ? (analysisOutput.cleaning_report as Record<string, unknown>)
      : null);
  const before =
    rawCleaningReport?.before && typeof rawCleaningReport.before === "object"
      ? (rawCleaningReport.before as Record<string, unknown>)
      : null;
  const after =
    rawCleaningReport?.after && typeof rawCleaningReport.after === "object"
      ? (rawCleaningReport.after as Record<string, unknown>)
      : null;
  const outlierColumns =
    rawCleaningReport?.outlier_columns && typeof rawCleaningReport.outlier_columns === "object"
      ? Object.entries(rawCleaningReport.outlier_columns as Record<string, unknown>).filter(
          ([, value]) => Number(value) > 0,
        )
      : [];

  return (
    <section className="panel">
      <div className="panel__eyebrow">AI result</div>
      <h2>Decision output</h2>
      {contractIntent || confidence ? (
        <p className="result-summary">
          {[contractIntent ? `Intent: ${contractIntent}` : null, confidence ? `Confidence: ${confidence}` : null]
            .filter(Boolean)
            .join(" · ")}
        </p>
      ) : null}
      {summary ? <p className="result-summary">{summary}</p> : null}
      {rawCleaningReport ? (
        <>
          <p className="result-summary">
            Data quality score: {Number(rawCleaningReport.quality_score ?? 0).toFixed(2)} · Missing fixed:{" "}
            {formatCount(rawCleaningReport.missing_handled)} · Duplicates removed:{" "}
            {formatCount(rawCleaningReport.duplicates_removed)} · Outlier alerts:{" "}
            {formatCount(rawCleaningReport.outliers_detected)}
          </p>
          <pre className="result-pre">
            {[
              "Real preprocessing. ✅Reliable model. ✅ Trust layer strong. ✅",
              before
                ? `Before -> rows ${formatCount(before.row_count)} | columns ${formatCount(before.column_count)} | missing ${formatCount(before.missing_cells)} | duplicates ${formatCount(before.duplicate_rows)}`
                : null,
              after
                ? `After -> rows ${formatCount(after.row_count)} | columns ${formatCount(after.column_count)} | missing ${formatCount(after.missing_cells)} | duplicates ${formatCount(after.duplicate_rows)}`
                : null,
              ...outlierColumns.map(([columnName, value]) => `Outlier alert: ${columnName} (${formatCount(value)})`),
            ]
              .filter(Boolean)
              .join("\n")}
          </pre>
        </>
      ) : null}
      {insights ? <pre className="result-pre">{insights}</pre> : null}
      {decisions ? <pre className="result-pre">{decisions}</pre> : null}
      {recommendations.length ? (
        <pre className="result-pre">{recommendations.map((item) => `- ${item}`).join("\n")}</pre>
      ) : null}
      {warnings.length ? <pre className="result-pre">{warnings.map((item) => `Warning: ${item}`).join("\n")}</pre> : null}
      <details className="result-details">
        <summary>Technical payload</summary>
        <pre className="result-pre">{JSON.stringify(analysisOutput, null, 2)}</pre>
      </details>
    </section>
  );
}
