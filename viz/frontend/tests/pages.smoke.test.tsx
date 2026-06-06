import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from '../src/App';

describe('App', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const okResponse = () =>
    new Response(
      JSON.stringify({ result: [], formula_latex: '', steps: [], source_code: '' }),
      { status: 200 }
    );

  it('shows the stage links and a collapsed Formulas group at the root', () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(okResponse());
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>
    );
    // Static stage headings + their links are always present.
    expect(screen.getAllByText(/Stage 1 Flow/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Stage 2a Flow/i).length).toBeGreaterThanOrEqual(1);
    // The Formulas group exists but is collapsed, so its demo links aren't rendered.
    expect(screen.getAllByText(/Formulas/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText(/One-Hot Truth/i)).toBeNull();
  });

  it('auto-expands the Formulas group when on a formula route', () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(okResponse());
    render(
      <MemoryRouter initialEntries={['/softmax']}>
        <App />
      </MemoryRouter>
    );
    for (const label of [
      /Softmax/i,
      /One-Hot Truth/i,
      /Cross-Entropy Loss/i,
      /Bigram Backward/i,
      /SGD Step/i,
      /Token Embedding/i,
      /Position Embedding/i,
      /Combined Input/i,
      /Lower-Tri Mask/i,
      /Normalize Rows/i,
      /Unembed Head/i,
    ]) {
      expect(screen.getAllByText(label).length).toBeGreaterThanOrEqual(1);
    }
  });

  const PAGE_PATHS = [
    '/softmax',
    '/one-hot-truth',
    '/cross-entropy-loss',
    '/bigram-backward',
    '/sgd-step',
    '/token-embed',
    '/position-embed',
    '/combined-input',
    '/lower-tri-mask',
    '/normalize-rows',
    '/unembed-head',
  ];

  it.each(PAGE_PATHS)('renders a legend on %s', async (path) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          result: [],
          formula_latex: '',
          steps: [],
          source_code: '',
        }),
        { status: 200 }
      )
    );
    const { container } = render(
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    );
    expect(container.querySelector('[data-testid="legend"]')).not.toBeNull();
  });
});
