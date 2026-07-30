// Viridis 5-stop ramp, interpolated per flood probability.
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],     // 0.0
  [59, 82, 139],   // 0.25
  [33, 145, 140],  // 0.5
  [94, 201, 98],   // 0.75
  [253, 231, 37],  // 1.0
];

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

export function viridis(t: number): [number, number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const scaled = x * (VIRIDIS.length - 1);
  const i = Math.floor(scaled);
  const j = Math.min(VIRIDIS.length - 1, i + 1);
  const f = scaled - i;
  const [r, g, b] = [0, 1, 2].map((k) => Math.round(lerp(VIRIDIS[i][k], VIRIDIS[j][k], f)));
  return [r, g, b, 200];
}
