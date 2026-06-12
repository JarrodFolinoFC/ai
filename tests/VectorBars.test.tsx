import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { VectorBars } from '../src/viz/VectorBars';

describe('VectorBars', () => {
  it('renders one bar per element', () => {
    const { container } = render(<VectorBars data={[1, -2, 3, 0.5]} />);
    expect(container.querySelectorAll('rect.bar')).toHaveLength(4);
  });

  it('renders an empty svg for an empty vector', () => {
    const { container } = render(<VectorBars data={[]} />);
    expect(container.querySelectorAll('rect.bar')).toHaveLength(0);
  });
});
