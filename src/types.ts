export type Vector = number[];
export type Matrix = number[][];
export type Scalar = number;
export type ResultValue = Scalar | Vector | Matrix;

export interface FormulaResponse {
  result: ResultValue;
  formula_latex: string;
  steps: [string, ResultValue][];
  source_code: string;
}

export interface TokenEmbedRequest {
  E: Matrix;
  idx: number;
}

export interface PositionEmbedRequest {
  P: Matrix;
  t: number;
}

export interface CombinedInputRequest {
  E: Matrix;
  P: Matrix;
  idx: number;
  t: number;
}

export interface LowerTriMaskRequest {
  T: number;
}

export interface NormalizeRowsRequest {
  M: Matrix;
}

export interface UnembedHeadRequest {
  h: Vector;
  W_head: Matrix;
  b: Vector;
}

export interface SoftmaxRequest {
  logits: Vector;
}

export interface CrossEntropyLossRequest {
  probs: Vector;
  target_id: number;
}

export interface BigramBackwardRequest {
  probs: Vector;
  target_id: number;
}

export interface SgdStepRequest {
  weight_row: Vector;
  grad_row: Vector;
  lr: number;
  n_examples: number;
}
