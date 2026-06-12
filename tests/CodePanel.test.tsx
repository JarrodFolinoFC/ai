import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { CodePanel } from '../src/components/CodePanel';

describe('CodePanel', () => {
  it('button is disabled when source is undefined', () => {
    render(<CodePanel source={undefined} />);
    const btn = screen.getByRole('button', { name: /show code/i });
    expect(btn).toBeDisabled();
  });

  it('toggles open and shows the source on click', () => {
    const src = 'def softmax(logits):\n    return logits';
    const { container } = render(<CodePanel source={src} />);
    const btn = screen.getByRole('button', { name: /show code/i });
    expect(btn).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('aria-expanded', 'true');
    // Prism wraps tokens in <span>s; check the rendered <code> contains the source.
    const code = container.querySelector('code.language-python');
    expect(code?.textContent).toContain('def softmax');
    expect(screen.getByRole('button', { name: /hide code/i })).toBeInTheDocument();
  });
});
