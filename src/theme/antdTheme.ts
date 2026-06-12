import type { ThemeConfig } from 'antd';

import { color } from './tokens';

// antd expects numeric px for radius/fontSize, while tokens.ts uses CSS strings
// for inline-style consumers. This module owns that conversion.
export const antdTheme: ThemeConfig = {
  token: {
    colorPrimary: color.highlight,
    colorSuccess: color.success.fg,
    colorWarning: color.warning.fg,
    colorError: color.error.fg,
    colorInfo: color.info.fg,
    colorText: color.text.primary,
    colorTextSecondary: color.text.secondary,
    colorBgContainer: color.bg.page,
    colorBgLayout: color.bg.surface,
    colorBorder: color.border.default,
    borderRadius: 4,
    fontSize: 16,
  },
};
