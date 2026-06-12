import { describe, expect, it } from 'vitest';

import { antdTheme } from '../src/theme';
import { color } from '../src/theme/tokens';

describe('antdTheme', () => {
  it('maps highlight to colorPrimary', () => {
    expect(antdTheme.token?.colorPrimary).toBe(color.highlight);
  });

  it('maps status token foregrounds', () => {
    expect(antdTheme.token?.colorSuccess).toBe(color.success.fg);
    expect(antdTheme.token?.colorWarning).toBe(color.warning.fg);
    expect(antdTheme.token?.colorError).toBe(color.error.fg);
    expect(antdTheme.token?.colorInfo).toBe(color.info.fg);
  });

  it('maps text, surface, and border tokens', () => {
    expect(antdTheme.token?.colorText).toBe(color.text.primary);
    expect(antdTheme.token?.colorTextSecondary).toBe(color.text.secondary);
    expect(antdTheme.token?.colorBgContainer).toBe(color.bg.page);
    expect(antdTheme.token?.colorBorder).toBe(color.border.default);
  });

  it('uses numeric radius and font size for antd', () => {
    expect(antdTheme.token?.borderRadius).toBe(4);
    expect(antdTheme.token?.fontSize).toBe(16);
  });
});
