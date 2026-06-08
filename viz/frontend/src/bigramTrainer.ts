import { bigramCounts, trainingPairs, trainToConvergence, randomMatrix, softmaxRow } from './funcs';
import type { Matrix } from './types';

export interface TrainingPair {
  prev: number;
  target: number;
}

// Snapshot of the most recently trained row, used to animate the before/after
// of a single step. null when not applicable (multi-step or not started).
export interface Flash {
  row: number;
  target: number;
  pairIdx: number;
  prevRow: number[];
}

// The next values for every piece of training state, ready to be destructured
// straight into the React setters.
export interface StepResult {
  W: Matrix;
  step: number;
  lossHistory: number[];
  flash: Flash | null;
}

// All values derived from the current state for the current render. Pure
// functions of (W, pairs, step, flash); destructured straight into the view.
export interface DerivedState {
  pairIdx: number;
  currentPair: TrainingPair;
  trainedRows: Set<number>;
  currentLogits: number[];
  currentExps: number[];
  currentExpSum: number;
  currentProbs: number[];
  currentLoss: number;
  prevLogits: number[];
  prevExps: number[];
  prevExpSum: number;
  prevProbs: number[];
  prevGrad: number[];
}

// Static properties of the corpus, independent of the live weights.
export interface CorpusStats {
  counts: Matrix;
  pairs: TrainingPair[];
  empirical: Matrix;
}

// The fully-trained reference model and the precomputed values its heatmaps need.
export interface Reference {
  wFinal: Matrix;
  softmaxWFinal: Matrix;
  wFinalMaxAbs: number;
}

// How far the live weights are from the empirical target, per row and overall.
export interface Analysis {
  softmaxW: Matrix;
  errorMatrix: Matrix;
  rowErrors: number[];
  totalError: number;
}

// Keep loss history bounded so it doesn't grow without limit during long runs.
const MAX_LOSS_HISTORY = 1000;

// Epochs used to train the reference (fully-converged) model.
const FINAL_EPOCHS = 500;

// Plain stateless trainer over a fixed bigram problem at a fixed position
// (W, pairs, step). `doStep` computes the next state; `derive` computes the
// read-only view for the current state. Pure: mutates nothing the caller passed
// in and performs no side effects.
export class BigramTrainer {
  constructor(
    private readonly W: Matrix,
    private readonly pairs: TrainingPair[],
    private readonly step: number
  ) {}

  // Counts, training pairs, and the empirical (target) distribution for a corpus.
  // Independent of the live weights, so callers memoize this on (corpus, vocab).
  static corpusStats(corpus: string, vocab: readonly string[]): CorpusStats {
    const { counts } = bigramCounts(corpus, vocab);
    const pairs = trainingPairs(corpus, vocab);
    const empirical = counts.map((row) => {
      const sum = row.reduce((a, b) => a + b, 0);
      return sum === 0 ? row.map(() => 0) : row.map((c) => c / sum);
    });
    return { counts, pairs, empirical };
  }

  // Train a reference model to convergence and precompute its derived views.
  // Expensive (FINAL_EPOCHS of SGD); callers memoize on (seed, vocab, pairs).
  static trainReference(
    seed: number,
    vocab: readonly string[],
    pairs: TrainingPair[],
    epochs = FINAL_EPOCHS
  ): Reference {
    const wFinal = trainToConvergence(
      randomMatrix(seed, vocab.length, vocab.length, 1),
      pairs,
      0.5,
      epochs
    );
    const softmaxWFinal = wFinal.map(softmaxRow);
    const wFinalMaxAbs = Math.max(...wFinal.flat().map((v) => Math.abs(v)), 1e-9);
    return { wFinal, softmaxWFinal, wFinalMaxAbs };
  }

  // Error of the live weights against the empirical target. Depends on W, so
  // callers memoize on (W, empirical).
  analysis(empirical: Matrix): Analysis {
    const softmaxW = this.W.map(softmaxRow);
    const errorMatrix = softmaxW.map((row, i) => row.map((p, j) => p - empirical[i][j]));
    const rowErrors = errorMatrix.map((row) => row.reduce((a, b) => a + Math.abs(b), 0) / 2);
    const totalError = rowErrors.reduce((a, b) => a + b, 0) / rowErrors.length;
    return { softmaxW, errorMatrix, rowErrors, totalError };
  }

  // Run `n` SGD steps from the current position and return the next state.
  doStep(n: number, lr: number, lossHistory: number[]): StepResult {
    const next = this.W.map((r) => r.slice());
    const newLosses: number[] = [];
    for (let k = 0; k < n; k++) {
      const pair = this.pairs[(this.step + k) % this.pairs.length];
      const row = next[pair.prev];
      const probs = softmaxRow(row);
      newLosses.push(-Math.log(Math.max(probs[pair.target], 1e-12)));
      const grad = probs.map((p, i) => (i === pair.target ? p - 1 : p));
      next[pair.prev] = row.map((v, i) => v - lr * grad[i]);
    }

    return {
      W: next,
      step: this.step + n,
      lossHistory: [...lossHistory, ...newLosses].slice(-MAX_LOSS_HISTORY),
      flash: n === 1 ? this.flashFor(this.step) : null,
    };
  }

  // Everything the view reads for the current state. `flash`, when present,
  // supplies the pre-step row so the forward pass can show before vs. after.
  derive(flash: Flash | null): DerivedState {
    const pairIdx = this.step === 0 ? 0 : (this.step - 1) % this.pairs.length;
    const currentPair = this.pairs[pairIdx];

    const trainedRows = new Set<number>();
    const trained = Math.min(this.step, this.pairs.length);
    for (let k = 0; k < trained; k++) trainedRows.add(this.pairs[k].prev);

    const currentLogits = this.W[currentPair.prev];
    const currentExps = currentLogits.map((v) => Math.exp(v));
    const currentExpSum = currentExps.reduce((a, b) => a + b, 0);
    const currentProbs = softmaxRow(currentLogits);
    const currentLoss = -Math.log(Math.max(currentProbs[currentPair.target], 1e-12));

    const prevLogits = flash ? flash.prevRow : currentLogits;
    const prevExps = prevLogits.map((v) => Math.exp(v));
    const prevExpSum = prevExps.reduce((a, b) => a + b, 0);
    const prevProbs = softmaxRow(prevLogits);
    const prevGrad = prevProbs.map((p, i) => (i === currentPair.target ? p - 1 : p));

    return {
      pairIdx,
      currentPair,
      trainedRows,
      currentLogits,
      currentExps,
      currentExpSum,
      currentProbs,
      currentLoss,
      prevLogits,
      prevExps,
      prevExpSum,
      prevProbs,
      prevGrad,
    };
  }

  // Before/after snapshot of the row trained at the given step.
  private flashFor(step: number): Flash {
    const pairIdx = step % this.pairs.length;
    const pair = this.pairs[pairIdx];
    return { row: pair.prev, target: pair.target, pairIdx, prevRow: this.W[pair.prev].slice() };
  }
}
