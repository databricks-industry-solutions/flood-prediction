import { useMemo, useState } from "react";
import { api, type AddressLookup } from "./api";
import { viridis } from "./colors";

type Props = {
  aoi: string;
  scenarioMm: number;
  onResult: (hit: AddressLookup | null) => void;
  result: AddressLookup | null;
};

function Sparkline({ sweep, currentMm }: { sweep: [number, number][]; currentMm: number }) {
  const w = 220;
  const h = 52;
  if (!sweep.length) return null;
  const xs = sweep.map(([mm]) => mm);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const points = sweep.map(([mm, p]) => {
    const x = ((mm - minX) / Math.max(1, maxX - minX)) * (w - 8) + 4;
    const y = h - 6 - p * (h - 12);
    return { mm, p, x, y };
  });
  const path = points.map((pt, i) => `${i ? "L" : "M"}${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} className="spark">
      <line x1={4} x2={w - 4} y1={h - 6} y2={h - 6} className="spark-axis" />
      <path d={path} className="spark-line" />
      {points.map((pt) => {
        const [r, g, b] = viridis(pt.p);
        const isActive = pt.mm === currentMm;
        return (
          <g key={pt.mm}>
            <circle
              cx={pt.x} cy={pt.y} r={isActive ? 4.5 : 2.5}
              fill={`rgb(${r},${g},${b})`}
              stroke={isActive ? "#e6edf3" : "none"}
              strokeWidth={isActive ? 1.5 : 0}
            />
            <text
              x={pt.x} y={h - 1}
              textAnchor="middle"
              className={isActive ? "spark-lbl active" : "spark-lbl"}
            >
              {pt.mm}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function AddressCard({ aoi, scenarioMm, onResult, result }: Props) {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (q.trim().length < 3) return;
    setLoading(true);
    setErr(null);
    try {
      const hit = await api.lookup(aoi, scenarioMm, q.trim());
      onResult(hit);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      onResult(null);
    } finally {
      setLoading(false);
    }
  }

  const prob = result ? result.flood_prob : 0;
  const [cr, cg, cb] = viridis(prob);
  const probColor = `rgb(${cr},${cg},${cb})`;

  const bars = useMemo(() => {
    if (!result) return [];
    return [
      { label: "Min elevation",    value: result.min_elev,        unit: "m",    norm: result.min_elev == null ? 0 : Math.max(0, 1 - result.min_elev / 80),   hi: "low" },
      { label: "Slope",            value: result.slope_deg,       unit: "°",    norm: result.slope_deg == null ? 0 : Math.max(0, 1 - result.slope_deg / 10), hi: "low" },
      { label: "Distance to water",value: result.dist_to_water_m, unit: "m",    norm: result.dist_to_water_m == null ? 0 : Math.max(0, 1 - result.dist_to_water_m / 1000), hi: "low" },
      { label: "Annual precip",    value: result.annual_precip_mm,unit: "mm",   norm: result.annual_precip_mm == null ? 0 : Math.min(1, result.annual_precip_mm / 1400),   hi: "hi"  },
    ];
  }, [result]);

  return (
    <div className="addr">
      <form onSubmit={submit} className="addr-form">
        <input
          type="text"
          placeholder="Check an address..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
          aria-label="Address search"
        />
        <button type="submit" disabled={loading || q.trim().length < 3}>
          {loading ? "..." : "Check"}
        </button>
      </form>
      {err && <div className="addr-err">{err}</div>}
      {result && (
        <div className="addr-card">
          <div className="addr-resolved" title={result.resolved_name}>
            {result.resolved_name}
          </div>
          <div className="addr-prob" style={{ color: probColor }}>
            {(prob * 100).toFixed(0)}
            <span className="addr-prob-unit">% flood risk</span>
          </div>
          <div className="addr-scenario">at {result.scenario_24h_mm} mm / 24 h</div>

          <div className="addr-bars">
            {bars.map((b) => (
              <div key={b.label} className="addr-bar">
                <div className="addr-bar-label">{b.label}</div>
                <div className="addr-bar-track">
                  <div
                    className="addr-bar-fill"
                    style={{ width: `${Math.round(b.norm * 100)}%` }}
                  />
                </div>
                <div className="addr-bar-value">
                  {b.value == null
                    ? "\u2014"
                    : `${b.unit === "°" ? b.value.toFixed(1) : Math.round(b.value)} ${b.unit}`}
                </div>
              </div>
            ))}
            {result.building_count != null && result.building_count > 0 && (
              <div className="addr-bar">
                <div className="addr-bar-label">Buildings in cell</div>
                <div className="addr-bar-track">
                  <div
                    className="addr-bar-fill"
                    style={{ width: `${Math.min(100, result.building_count * 5)}%` }}
                  />
                </div>
                <div className="addr-bar-value">{result.building_count}</div>
              </div>
            )}
          </div>

          <div className="addr-spark-title">Risk across rainfall scenarios</div>
          <Sparkline sweep={result.sweep} currentMm={result.scenario_24h_mm} />

          <button type="button" className="addr-clear" onClick={() => onResult(null)}>
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
