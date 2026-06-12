import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { VectorEditor } from '../src/components/VectorEditor';

describe('VectorEditor', () => {
  it('renders one labelled input per cell', () => {
    render(<VectorEditor value={[1, 2, 3]} onChange={() => {}} label="v" />);
    expect(screen.getByLabelText('v[0]')).toBeInTheDocument();
    expect(screen.getByLabelText('v[1]')).toBeInTheDocument();
    expect(screen.getByLabelText('v[2]')).toBeInTheDocument();
  });

  it('calls onChange with the updated vector when a cell changes', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<VectorEditor value={[1, 2, 3]} onChange={onChange} label="v" />);
    const cell = screen.getByLabelText('v[1]');
    await user.clear(cell);
    await user.type(cell, '9');
    expect(onChange).toHaveBeenCalled();
    const last = onChange.mock.calls.at(-1)![0];
    expect(last[1]).toBe(9);
  });
});
