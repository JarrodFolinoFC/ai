import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { MatrixGrid } from '../src/viz/MatrixGrid';

describe('MatrixGrid', () => {
  it('renders one cell per matrix entry', () => {
    const { container } = render(
      <MatrixGrid data={[[1, 2, 3], [4, 5, 6]]} />
    );
    expect(container.querySelectorAll('rect.cell')).toHaveLength(6);
  });

  it('marks the highlighted row', () => {
    const { container } = render(
      <MatrixGrid data={[[1, 2], [3, 4]]} highlightRow={1} />
    );
    expect(container.querySelectorAll('rect.cell.highlighted')).toHaveLength(2);
  });
});
