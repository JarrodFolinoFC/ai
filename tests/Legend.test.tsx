import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Legend } from '../src/components/Legend';

describe('Legend', () => {
  it('renders one row per entry', () => {
    const { container } = render(
      <Legend
        entries={[
          { symbol: 'z', meaning: 'logits vector' },
          { symbol: '\\sigma(z)_i', meaning: 'softmax probability' },
        ]}
      />
    );
    expect(container.querySelectorAll('[data-testid="legend-row"]')).toHaveLength(
      2
    );
  });

  it('renders each symbol with KaTeX', () => {
    const { container } = render(
      <Legend entries={[{ symbol: 'z_i', meaning: 'i-th logit' }]} />
    );
    const row = container.querySelector('[data-testid="legend-row"]');
    expect(row?.querySelector('.katex')).not.toBeNull();
  });

  it('renders the meaning text verbatim', () => {
    const { container } = render(
      <Legend
        entries={[{ symbol: 'z', meaning: 'logits vector (raw scores)' }]}
      />
    );
    expect(container.textContent).toContain('logits vector (raw scores)');
  });

  it('renders nothing visible for empty entries', () => {
    const { container } = render(<Legend entries={[]} />);
    expect(container.querySelectorAll('[data-testid="legend-row"]')).toHaveLength(
      0
    );
  });
});
