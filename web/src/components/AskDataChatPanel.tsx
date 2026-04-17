import { useEffect, useState } from "react";
import { Bot, Loader2, SendHorizontal, Sparkles } from "lucide-react";

import { askYourData } from "../lib/api";
import type { AskDataResponse, InsightChart } from "../types/api";

type Message = {
  role: "user" | "assistant";
  text: string;
  sql?: string;
  rows?: Record<string, unknown>[];
  columns?: string[];
  chart?: InsightChart | null;
};

type AskDataChatPanelProps = {
  assetId: string | null;
  token: string | null;
  suggestedQuestions?: string[];
  initialQuestion?: string | null;
  onQuestionConsumed?: () => void;
};

function toNumeric(value: unknown): number {
  return typeof value === "number" ? value : Number(value || 0);
}

function ResultChart({ chart }: { chart?: InsightChart | null }) {
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

function buildAssistantMessage(payload: AskDataResponse): Message {
  return {
    role: "assistant",
    text: payload.answer,
    sql: payload.sql,
    rows: payload.rows,
    columns: payload.columns,
    chart: payload.chart
  };
}

export default function AskDataChatPanel({
  assetId,
  token,
  suggestedQuestions = [],
  initialQuestion,
  onQuestionConsumed
}: AskDataChatPanelProps) {
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Ask about revenue drops, top customers, campaign performance, or trend changes. Aidssist will translate the question into SQL and show the result."
    }
  ]);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialQuestion) {
      setDraft(initialQuestion);
      onQuestionConsumed?.();
    }
  }, [initialQuestion, onQuestionConsumed]);

  async function submitQuestion(question: string) {
    if (!assetId || !token || !question.trim()) {
      return;
    }
    const resolvedQuestion = question.trim();
    setWorking(true);
    setError(null);
    setMessages((existing) => [...existing, { role: "user", text: resolvedQuestion }]);
    setDraft("");
    try {
      const payload = await askYourData({ asset_id: assetId, question: resolvedQuestion }, token);
      setMessages((existing) => [...existing, buildAssistantMessage(payload)]);
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : "Ask Your Data failed.";
      setError(message);
      setMessages((existing) => [
        ...existing,
        {
          role: "assistant",
          text: message
        }
      ]);
    } finally {
      setWorking(false);
    }
  }

  return (
    <article className="surface-card surface-card--wide ask-data-shell">
      <div className="ask-data-shell__header">
        <div>
          <span className="section-kicker">AI Assistant Panel</span>
          <h3>Ask Your Data</h3>
          <p>Natural-language questions become safe DuckDB SQL queries, plus an answer and chart-ready output.</p>
        </div>
        <div className="import-hub-card__pill">
          <Bot size={18} />
          <span>Chat + SQL</span>
        </div>
      </div>

      {suggestedQuestions.length ? (
        <div className="question-chip-row">
          {suggestedQuestions.slice(0, 5).map((question) => (
            <button key={question} type="button" className="question-chip" onClick={() => void submitQuestion(question)}>
              <Sparkles size={13} />
              {question}
            </button>
          ))}
        </div>
      ) : null}

      <div className="chat-thread">
        {messages.map((message, index) => (
          <article
            key={`${message.role}-${index}-${message.text.slice(0, 24)}`}
            className={`chat-message ${message.role === "assistant" ? "chat-message--assistant" : "chat-message--user"}`}
          >
            <div className="chat-message__meta">{message.role === "assistant" ? "Aidssist AI" : "You"}</div>
            <p>{message.text}</p>
            {message.sql ? <pre className="chat-sql-block">{message.sql}</pre> : null}
            {message.chart ? <ResultChart chart={message.chart} /> : null}
            {message.rows?.length ? (
              <div className="table-shell chat-result-table">
                <table>
                  <thead>
                    <tr>
                      {(message.columns || []).map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {message.rows.slice(0, 6).map((row, rowIndex) => (
                      <tr key={`${index}-${rowIndex}`}>
                        {(message.columns || []).map((column) => (
                          <td key={`${column}-${rowIndex}`}>{String(row[column] ?? "")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </article>
        ))}
      </div>

      <form
        className="ask-data-composer"
        onSubmit={(event) => {
          event.preventDefault();
          void submitQuestion(draft);
        }}
      >
        <textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={assetId ? 'Example: "Why did revenue drop last month?"' : "Select an asset to start chatting with your data"}
          disabled={!assetId || working}
        />
        <button type="submit" className="primary-button-new" disabled={!assetId || !draft.trim() || working}>
          {working ? (
            <>
              <Loader2 size={16} className="spin" />
              Thinking...
            </>
          ) : (
            <>
              <SendHorizontal size={16} />
              Ask
            </>
          )}
        </button>
      </form>

      {error ? <div className="notice-card notice-card--error">{error}</div> : null}
    </article>
  );
}
