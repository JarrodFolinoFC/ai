export function bigramCounts(corpus: string, vocab: readonly string[]) {
  const idx = new Map(vocab.map((w, i) => [w, i] as const));
  const tokens = corpus.split(/\s+/).filter(Boolean);
  const V = vocab.length;
  const counts = Array.from({ length: V }, () => Array<number>(V).fill(0));
  for (let i = 0; i < tokens.length - 1; i++) {
    const a = idx.get(tokens[i]);
    const b = idx.get(tokens[i + 1]);
    if (a === undefined || b === undefined) continue;
    counts[a][b] += 1;
  }
  return { counts, total: tokens.length };
}
export function trainingPairs(corpus: string, vocab: readonly string[]) {
  const idx = new Map(vocab.map((w, i) => [w, i] as const));
  const tokens = corpus.split(/\s+/).filter(Boolean);
  const pairs: { prev: number; target: number }[] = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    const a = idx.get(tokens[i]);
    const b = idx.get(tokens[i + 1]);
    if (a === undefined || b === undefined) continue;
    pairs.push({ prev: a, target: b });
  }
  return pairs;
}
export function trainToConvergence(
  initialW: number[][],
  pairs: { prev: number; target: number }[],
  lr: number,
  epochs: number
) {
  const W = initialW.map((r) => r.slice());
  for (let e = 0; e < epochs; e++) {
    for (const pair of pairs) {
      const row = W[pair.prev];
      const probs = softmaxRow(row);
      const grad = probs.map((p, i) => (i === pair.target ? p - 1 : p));
      W[pair.prev] = row.map((v, i) => v - lr * grad[i]);
    }
  }
  return W;
}
export function randomMatrix(
  seed: number,
  rows: number,
  cols: number,
  scale = 1
) {
  const rand = mulberry32(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (rand() - 0.5) * 2 * scale)
  );
}

export function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function tokenize(corpus: string) {
  return corpus.split(/\s+/).filter(Boolean);
}

export function tokenIds(tokens: string[], vocab: readonly string[]) {
  const idx = new Map(vocab.map((w, i) => [w, i] as const));
  return tokens.map((tok) => idx.get(tok) ?? -1);
}

export function lowerTriMask(T: number) {
  return Array.from({ length: T }, (_, i) =>
    Array.from({ length: T }, (_, j) => (j <= i ? 1 : 0))
  );
}

export function sinusoidalP(T: number, dModel: number) {
  const P: number[][] = [];
  for (let pos = 0; pos < T; pos++) {
    const row: number[] = [];
    for (let i = 0; i < dModel; i++) {
      const evenI = i % 2 === 0 ? i : i - 1;
      const denom = Math.pow(10000, evenI / dModel);
      row.push(i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom));
    }
    P.push(row);
  }
  return P;
}
export function rowNormalize(M: number[][]) {
  return M.map((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    return sum > 0 ? row.map((v) => v / sum) : row.slice();
  });
}

export function matMul(A: number[][], B: number[][]) {
  const rows = A.length;
  const inner = B.length;
  const cols = B[0].length;
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => {
      let acc = 0;
      for (let k = 0; k < inner; k++) acc += A[i][k] * B[k][j];
      return acc;
    })
  );
}

// Mask-aware: non-finite entries (e.g. -Infinity from causal masking) contribute
// zero probability. Identical to a plain softmax for all-finite rows.
export function softmaxRow(row: number[]) {
  const finite = row.filter((v) => Number.isFinite(v));
  const max = finite.length ? Math.max(...finite) : 0;
  const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => (sum > 0 ? e / sum : 0));
}

export function addVec(a: number[], b: number[]) {
  return a.map((x, i) => x + b[i]);
}

export function scaleVec(a: number[], s: number) {
  return a.map((x) => x * s);
}

export function zeros(n: number) {
  return Array<number>(n).fill(0);
}

export function transpose(M: number[][]) {
  const rows = M.length;
  const cols = M[0].length;
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => M[i][j])
  );
}

// W^T · v + b : right-multiply a (rows × cols) matrix's columns by vector v.
export function matVecRight(W: number[][], v: number[], b: number[]) {
  const cols = W[0].length;
  return Array.from(
    { length: cols },
    (_, j) => v.reduce((acc, vk, k) => acc + vk * W[k][j], 0) + b[j]
  );
}

export function maxAbs(matrix: number[][]) {
  return Math.max(...matrix.flat().map((v) => Math.abs(v)), 1e-9);
}

// Split [0, total) into contiguous index chunks of at most `size` (table wrapping).
export function chunkIndices(total: number, size: number) {
  const out: number[][] = [];
  for (let start = 0; start < total; start += size) {
    const end = Math.min(start + size, total);
    const chunk: number[] = [];
    for (let i = start; i < end; i++) chunk.push(i);
    out.push(chunk);
  }
  return out;
}
