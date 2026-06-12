import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '../src/api';

describe('api.tokenEmbed', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('POSTs to /api/stage2a/token-embed and returns parsed result', async () => {
    const mockResponse = {
      result: [1.0, 2.0],
      formula_latex: 'x = E[idx]',
      steps: [['row', [1.0, 2.0]]],
    };
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(mockResponse), { status: 200 })
    );
    const out = await api.tokenEmbed({ E: [[0.0, 1.0], [1.0, 2.0]], idx: 1 });
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/stage2a/token-embed');
    expect(init?.method).toBe('POST');
    expect(out.result).toEqual([1.0, 2.0]);
  });

  it('throws ApiError on non-2xx', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'bad' }), { status: 422 })
    );
    await expect(
      api.tokenEmbed({ E: [[0.0]], idx: 99 })
    ).rejects.toThrow(/422/);
  });
});

describe('api stage 1 methods', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const mockOk = (body: unknown) =>
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(body), { status: 200 })
    );

  it('softmax POSTs /api/stage1/softmax', async () => {
    const fetchMock = mockOk({
      result: [0.5, 0.5],
      formula_latex: '',
      steps: [],
      source_code: 'def softmax(...): ...',
    });
    await api.softmax({ logits: [0, 0] });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/stage1/softmax');
  });

  it('crossEntropyLoss POSTs /api/stage1/cross-entropy-loss', async () => {
    const fetchMock = mockOk({
      result: 0.5,
      formula_latex: '',
      steps: [],
      source_code: 'def cross_entropy_loss(...): ...',
    });
    await api.crossEntropyLoss({ probs: [0.5, 0.5], target_id: 0 });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/stage1/cross-entropy-loss');
  });

  it('bigramBackward POSTs /api/stage1/bigram-backward', async () => {
    const fetchMock = mockOk({
      result: [0.5, -0.5],
      formula_latex: '',
      steps: [],
      source_code: 'def bigram_backward(...): ...',
    });
    await api.bigramBackward({ probs: [0.5, 0.5], target_id: 1 });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/stage1/bigram-backward');
  });

  it('sgdStep POSTs /api/stage1/sgd-step', async () => {
    const fetchMock = mockOk({
      result: [0.95, 1.05],
      formula_latex: '',
      steps: [],
      source_code: 'def sgd_step(...): ...',
    });
    await api.sgdStep({
      weight_row: [1, 1],
      grad_row: [1, -1],
      lr: 0.1,
      n_examples: 2,
    });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/stage1/sgd-step');
  });
});
