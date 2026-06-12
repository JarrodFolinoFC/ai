import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from '../src/App';

function mockFetchOk() {
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
}

describe('CombinedInputPage', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders corpus size picker, d_model dropdown, and seed input', async () => {
    mockFetchOk();
    await act(async () => {
      render(
        <MemoryRouter initialEntries={['/combined-input']}>
          <App />
        </MemoryRouter>
      );
    });
    // Corpus size radios (25, 50, 75) from the standardized TrainingCorpus block.
    expect(screen.getByLabelText(/^\s*25/)).toBeInTheDocument();
    expect(screen.getByLabelText(/^\s*50/)).toBeInTheDocument();
    expect(screen.getByLabelText(/^\s*75/)).toBeInTheDocument();
    // d_model dropdown
    expect(screen.getByLabelText(/d_model/i)).toBeInTheDocument();
    // seed input
    expect(screen.getByLabelText(/seed/i)).toBeInTheDocument();
  });

  it('renders the four corpus matrices (E vocab toggle, E[idx_t] toggle, P[t], h)', async () => {
    mockFetchOk();
    await act(async () => {
      render(
        <MemoryRouter initialEntries={['/combined-input']}>
          <App />
        </MemoryRouter>
      );
    });
    // Default E view is 'vocab'; the lookup view radio is also rendered as a label.
    expect(screen.getByTestId('matrix-e-vocab')).toBeInTheDocument();
    expect(screen.getByTestId('matrix-p')).toBeInTheDocument();
    expect(screen.getByTestId('matrix-h')).toBeInTheDocument();
  });

  it('changes column count when d_model dropdown changes', async () => {
    mockFetchOk();
    const user = userEvent.setup();
    await act(async () => {
      render(
        <MemoryRouter initialEntries={['/combined-input']}>
          <App />
        </MemoryRouter>
      );
    });

    // 3 tables are rendered (E vocab OR E[idx_t] — toggled, default 'vocab' — plus h and P).
    // Each has 8 numeric column headers; the last column index is "7".
    expect(screen.getAllByRole('columnheader', { name: '7' })).toHaveLength(3);
    expect(screen.queryAllByRole('columnheader', { name: '15' })).toHaveLength(0);

    await user.click(screen.getByLabelText(/d_model/i));
    await user.click(await screen.findByTitle('16'));

    // Now there should be a "15" column header in each of the 3 tables.
    expect(screen.getAllByRole('columnheader', { name: '15' })).toHaveLength(3);
  });

  it('produces identical E values across renders for the same seed and d_model', async () => {
    mockFetchOk();
    let unmount!: () => void;
    await act(async () => {
      const result = render(
        <MemoryRouter initialEntries={['/combined-input']}>
          <App />
        </MemoryRouter>
      );
      unmount = result.unmount;
    });
    // Capture the first cell of E[idx] for the first vocab row.
    // E[0][0] lives in the E vocab table, row idx=0, dim 0.
    const eTable = screen.getByTestId('matrix-e-vocab').querySelector('table');
    expect(eTable).not.toBeNull();
    const firstRowCells = eTable!.querySelectorAll('tbody tr:first-child td');
    // first <td> in the first <tr> is dim 0 (the two leading <th>s are idx and token)
    const firstRender = firstRowCells[0].textContent;

    unmount();

    // Render again with same defaults → same seed + d_model
    await act(async () => {
      render(
        <MemoryRouter initialEntries={['/combined-input']}>
          <App />
        </MemoryRouter>
      );
    });
    const eTable2 = screen.getByTestId('matrix-e-vocab').querySelector('table');
    expect(eTable2).not.toBeNull();
    const firstRowCells2 = eTable2!.querySelectorAll('tbody tr:first-child td');
    expect(firstRowCells2[0].textContent).toBe(firstRender);
  });
});
