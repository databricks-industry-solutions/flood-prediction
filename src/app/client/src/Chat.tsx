import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { api, type ChatMessage } from "./api";

type Turn =
  | { role: "user"; content: string; ts: number }
  | { role: "assistant"; msg: ChatMessage; ts: number };

type Props = {
  aoi: string;
  scenarioMm: number;
};

const SAMPLE_QUESTIONS = [
  "What is the total expected loss at the current scenario?",
  "Top 20 buildings by expected loss in this AOI",
  "Residential vs commercial expected loss breakdown",
  "How many buildings are in each risk tier?",
  "Compare aggregate expected loss across all scenarios",
  "How many severe-risk buildings sit inside the 2017 flood polygon?",
];

const POLL_INTERVAL_MS = 1200;
const POLL_TIMEOUT_MS = 60_000;
const STORAGE_KEY = "flood-chat-v1";

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
    if (Math.abs(v) >= 1_000)     return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (Number.isInteger(v))      return v.toString();
    return v.toFixed(3);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

function isSingleScalar(msg: ChatMessage): boolean {
  return !!(
    msg.rows && msg.rows.length === 1
    && msg.columns && msg.columns.length === 1
    && typeof msg.rows[0][0] === "number"
  );
}

function loadStored(): Turn[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { turns?: Turn[] };
    return Array.isArray(parsed.turns) ? parsed.turns : [];
  } catch {
    return [];
  }
}

function loadConvId(): string | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { conversationId?: string };
    return parsed.conversationId ?? null;
  } catch {
    return null;
  }
}

export function Chat({ aoi, scenarioMm }: Props) {
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [open, setOpen] = useState(true);
  const [turns, setTurns] = useState<Turn[]>(() => loadStored());
  const [conversationId, setConversationId] = useState<string | null>(() => loadConvId());
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [pollingFor, setPollingFor] = useState<string | null>(null);
  const [pollElapsed, setPollElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const scrollerRef = useRef<HTMLDivElement | null>(null);

  // Health-check once on mount; hides the panel entirely if Genie isn't wired.
  useEffect(() => {
    api.chatHealth()
      .then((h) => setEnabled(h.enabled))
      .catch(() => setEnabled(false));
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ turns, conversationId }));
  }, [turns, conversationId]);

  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, pollingFor]);

  const pollUntilDone = useCallback(
    async (convId: string, msgId: string): Promise<ChatMessage> => {
      const start = Date.now();
      setPollingFor(msgId);
      try {
        while (Date.now() - start < POLL_TIMEOUT_MS) {
          setPollElapsed(Math.round((Date.now() - start) / 1000));
          const m = await api.chatPoll(convId, msgId);
          if (m.status !== "PENDING") return m;
          await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        }
        return {
          conversation_id: convId,
          message_id: msgId,
          status: "FAILED",
          text: null, sql: null, columns: null, rows: null, row_count: null,
          truncated: false,
          error: `Genie did not respond within ${POLL_TIMEOUT_MS / 1000}s`,
        };
      } finally {
        setPollingFor(null);
        setPollElapsed(0);
      }
    },
    [],
  );

  const send = useCallback(
    async (content: string) => {
      if (!content.trim() || sending) return;
      setError(null);
      setSending(true);
      const userTurn: Turn = { role: "user", content, ts: Date.now() };
      setTurns((prev) => [...prev, userTurn]);
      setInput("");
      try {
        let initial: ChatMessage;
        if (!conversationId) {
          initial = await api.chatStart(content, aoi, scenarioMm);
          setConversationId(initial.conversation_id);
        } else {
          initial = await api.chatSend(conversationId, content, aoi, scenarioMm);
        }
        const final =
          initial.status === "PENDING"
            ? await pollUntilDone(initial.conversation_id, initial.message_id)
            : initial;
        setTurns((prev) => [...prev, { role: "assistant", msg: final, ts: Date.now() }]);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      } finally {
        setSending(false);
      }
    },
    [aoi, scenarioMm, conversationId, sending, pollUntilDone],
  );

  const reset = useCallback(() => {
    setTurns([]);
    setConversationId(null);
    setError(null);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void send(input);
      }
    },
    [input, send],
  );

  const hasHistory = turns.length > 0;
  const contextLabel = useMemo(
    () => `Asking with: AOI=${aoi}, scenario=${scenarioMm} mm`,
    [aoi, scenarioMm],
  );

  if (enabled === null) return null;     // probing
  if (enabled === false) return null;    // Genie not wired in this deploy

  return (
    <div className={`chat ${open ? "chat-open" : "chat-closed"}`}>
      <button
        type="button"
        className="chat-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="chat-toggle-title">
          <ChatLogo />
          <span className="chat-toggle-label">
            <span className="chat-toggle-name">Underwriter chat</span>
            <span className="chat-toggle-sub">Ask about portfolio exposure</span>
          </span>
        </span>
        <span className="chat-chev" aria-hidden>{open ? "\u2013" : "+"}</span>
      </button>
      {open && (
        <div className="chat-body">
          <div className="chat-context">{contextLabel}</div>
          <div className="chat-scroll" ref={scrollerRef}>
            {!hasHistory && (
              <div className="chat-empty">
                <div className="chat-empty-title">Ask about portfolio exposure.</div>
                <ul className="chat-samples">
                  {SAMPLE_QUESTIONS.map((q) => (
                    <li key={q}>
                      <button
                        type="button"
                        className="chat-sample"
                        onClick={() => void send(q)}
                        disabled={sending}
                      >
                        {q}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {turns.map((t, i) => (t.role === "user"
              ? <UserBubble key={i} content={t.content} />
              : <AssistantBubble key={i} msg={t.msg} />
            ))}
            {pollingFor && (
              <div className="chat-pending">
                Genie is thinking... <span className="chat-pending-secs">{pollElapsed}s</span>
              </div>
            )}
            {error && <div className="chat-error">{error}</div>}
          </div>
          <div className="chat-input-row">
            <textarea
              className="chat-input"
              placeholder="Ask about expected loss, top-N exposures, scenario stress..."
              rows={2}
              value={input}
              disabled={sending}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <div className="chat-input-actions">
              <button
                type="button"
                className="chat-send"
                onClick={() => void send(input)}
                disabled={sending || !input.trim()}
              >
                {sending ? "..." : "Send"}
              </button>
              {hasHistory && (
                <button
                  type="button"
                  className="chat-reset"
                  onClick={reset}
                  disabled={sending}
                >
                  Reset
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ChatLogo() {
  // Compact 32px badge: a circular gradient (Databricks-style teal -> blue)
  // holding a chat bubble whose tail dips into a stylized wave - underwriter
  // chat + flood risk in one glance. Inline SVG so it scales, themes via
  // currentColor, and ships with no extra HTTP request.
  return (
    <svg
      className="chat-logo"
      width="32"
      height="32"
      viewBox="0 0 32 32"
      role="img"
      aria-label="Underwriter chat logo"
    >
      <defs>
        <linearGradient id="chatLogoBg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"  stopColor="#21918c" />
          <stop offset="60%" stopColor="#1f6feb" />
          <stop offset="100%" stopColor="#0d3b66" />
        </linearGradient>
        <linearGradient id="chatLogoWave" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"  stopColor="#9be7ff" />
          <stop offset="100%" stopColor="#58a6ff" />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="15" fill="url(#chatLogoBg)" stroke="#0b0f14" strokeWidth="1" />
      {/* Chat bubble */}
      <path
        d="M9 11.5 a3.5 3.5 0 0 1 3.5 -3.5 h7 a3.5 3.5 0 0 1 3.5 3.5 v4.5 a3.5 3.5 0 0 1 -3.5 3.5 h-5.2 l-2.6 2.5 v-2.5 h-1.2 a3.5 3.5 0 0 1 -3.5 -3.5 z"
        fill="#0d1117"
        stroke="#e6edf3"
        strokeWidth="1.1"
        strokeLinejoin="round"
      />
      {/* Wave inside bubble */}
      <path
        d="M11.5 14.4 c1 -1 2 -1 3 0 s2 1 3 0 s2 -1 3 0"
        fill="none"
        stroke="url(#chatLogoWave)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      {/* AI sparkle */}
      <circle cx="22.5" cy="10.5" r="1.1" fill="#fde725" />
    </svg>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="chat-msg chat-msg-user">
      <div className="chat-bubble">{content}</div>
    </div>
  );
}

function AssistantBubble({ msg }: { msg: ChatMessage }) {
  const single = isSingleScalar(msg);
  return (
    <div className="chat-msg chat-msg-assistant">
      <div className="chat-bubble">
        {msg.status === "FAILED" && (
          <div className="chat-err-line">{msg.error ?? "Genie failed."}</div>
        )}
        {msg.text && <div className="chat-text">{msg.text}</div>}
        {single && msg.rows && msg.columns && (
          <div className="chat-bignum">
            <div className="chat-bignum-v">{formatCell(msg.rows[0][0])}</div>
            <div className="chat-bignum-l">{msg.columns[0]}</div>
          </div>
        )}
        {!single && msg.rows && msg.columns && msg.rows.length > 0 && (
          <ResultTable columns={msg.columns} rows={msg.rows} truncated={msg.truncated} />
        )}
        {msg.sql && (
          <details className="chat-sql">
            <summary>SQL ({msg.row_count ?? msg.rows?.length ?? 0} rows)</summary>
            <pre>{msg.sql}</pre>
          </details>
        )}
      </div>
    </div>
  );
}

function ResultTable({
  columns,
  rows,
  truncated,
}: {
  columns: string[];
  rows: (string | number | boolean | null)[][];
  truncated: boolean;
}) {
  const [showAll, setShowAll] = useState(false);
  const limit = 12;
  const visible = showAll ? rows : rows.slice(0, limit);
  const more = rows.length - visible.length;

  return (
    <div className="chat-result">
      <div className="chat-result-scroll">
        <table>
          <thead>
            <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {visible.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} className={typeof cell === "number" ? "num" : ""}>
                    {formatCell(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(more > 0 || truncated) && (
        <div className="chat-result-foot">
          {more > 0 && (
            <button
              type="button"
              className="chat-show-all"
              onClick={() => setShowAll((v) => !v)}
            >
              {showAll ? "Show fewer" : `Show ${more} more`}
            </button>
          )}
          {truncated && <span className="chat-result-truncated">truncated by server</span>}
        </div>
      )}
    </div>
  );
}
