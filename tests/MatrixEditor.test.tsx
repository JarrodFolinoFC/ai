import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { MatrixEditor } from '../src/components/MatrixEditor';

describe('MatrixEditor', () => {
  it('renders one labelled input per cell', () => {
    render(<MatrixEditor value={[[1, 2], [3, 4]]} onChange={() => {}} label="m" />);
    expect(screen.getByLabelText('m[0][0]')).toBeInTheDocument();
    expect(screen.getByLabelText('m[1][1]')).toBeInTheDocument();
  });

  it('calls onChange with the updated matrix when a cell changes', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<MatrixEditor value={[[1, 2], [3, 4]]} onChange={onChange} label="m" />);
    const cell = screen.getByLabelText('m[1][0]');
    await user.clear(cell);
    await user.type(cell, '7');
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls.at(-1)![0];
    expect(last[1][0]).toBe(7);
  });
});
