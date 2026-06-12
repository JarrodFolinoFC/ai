import type {
  BigramBackwardRequest,
  CombinedInputRequest,
  CrossEntropyLossRequest,
  FormulaResponse,
  LowerTriMaskRequest,
  NormalizeRowsRequest,
  PositionEmbedRequest,
  SgdStepRequest,
  SoftmaxRequest,
  TokenEmbedRequest,
  UnembedHeadRequest,
} from './types';

export class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(`API ${status}: ${detail}`);
  }
}

async function post<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = (await res.json()) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      // body wasn't JSON; fall through with statusText
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<TRes>;
}

export const api = {
  tokenEmbed: (req: TokenEmbedRequest) =>
    post<TokenEmbedRequest, FormulaResponse>('/api/stage2a/token-embed', req),
  positionEmbed: (req: PositionEmbedRequest) =>
    post<PositionEmbedRequest, FormulaResponse>('/api/stage2a/position-embed', req),
  combinedInput: (req: CombinedInputRequest) =>
    post<CombinedInputRequest, FormulaResponse>('/api/stage2a/combined-input', req),
  lowerTriMask: (req: LowerTriMaskRequest) =>
    post<LowerTriMaskRequest, FormulaResponse>('/api/stage2a/lower-tri-mask', req),
  normalizeRows: (req: NormalizeRowsRequest) =>
    post<NormalizeRowsRequest, FormulaResponse>('/api/stage2a/normalize-rows', req),
  unembedHead: (req: UnembedHeadRequest) =>
    post<UnembedHeadRequest, FormulaResponse>('/api/stage2a/unembed-head', req),
  softmax: (req: SoftmaxRequest) =>
    post<SoftmaxRequest, FormulaResponse>('/api/stage1/softmax', req),
  crossEntropyLoss: (req: CrossEntropyLossRequest) =>
    post<CrossEntropyLossRequest, FormulaResponse>('/api/stage1/cross-entropy-loss', req),
  bigramBackward: (req: BigramBackwardRequest) =>
    post<BigramBackwardRequest, FormulaResponse>('/api/stage1/bigram-backward', req),
  sgdStep: (req: SgdStepRequest) =>
    post<SgdStepRequest, FormulaResponse>('/api/stage1/sgd-step', req),
};
