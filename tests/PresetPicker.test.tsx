import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { PresetPicker } from '../src/components/PresetPicker';

describe('PresetPicker', () => {
  it('calls onPick with the chosen preset value', async () => {
    const user = userEvent.setup();
    const onPick = vi.fn();
    render(
      <PresetPicker
        presets={[
          { label: 'Alpha', value: 1 },
          { label: 'Beta', value: 2 },
        ]}
        onPick={onPick}
      />
    );
    // Open the combobox and choose "Beta".
    await user.click(screen.getByRole('combobox'));
    await user.click(await screen.findByText('Beta'));
    expect(onPick).toHaveBeenCalledWith(2);
  });
});
