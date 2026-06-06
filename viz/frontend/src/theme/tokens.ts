// Single source of truth for design tokens. No framework imports — these plain
// values are consumed by antd (via antdTheme.ts), inline styles, and d3.

export const color = {
  text: {
    primary: '#0f172a', // absorbs near-black navies (#111827, #1f2937, #1e293b) and #000
    secondary: '#555', //   absorbs #444, #666, #6b7280
    muted: '#9ca3af', //     absorbs #888, #aaa
    inverse: '#fff', //      light text/fill on dark fills
    onDark: '#e2e8f0', //    light caption text on dark panels
    emphasis: '#0c4a6e', //  dark-blue marked / target text
  },
  bg: {
    page: '#fff',
    surface: '#fafafa',
    subtle: '#f6f8fa',
    disabled: '#e5e7eb', //  struck / disabled cell background
  },
  border: {
    default: '#ddd',
    strong: '#ccc',
  },

  // signed data values (d3 bars, deltas)
  positive: '#2a7', //  absorbs #34d399
  negative: '#d35',

  // callout / status families
  info: { fg: '#0369a1', bg: '#f0f9ff', border: '#bae6fd' },
  warning: { fg: '#92400e', bg: '#fffbeb', border: '#fcd34d' },
  success: { fg: '#15803d', bg: '#dcfce7', border: '#bbf7d0' },
  error: { fg: '#b91c1c', bg: '#fee2e2', border: '#fecaca' },

  // selected cell / emphasis (amber)
  highlight: '#f59e0b', // solid amber border / accent
  highlightBg: '#fde68a', // highlighted cell / row background
} as const;

export const space = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '0.75rem',
  lg: '1rem',
  xl: '1.5rem',
  xxl: '2.5rem',
} as const;

export const radius = {
  sm: '4px',
} as const;

export const font = {
  size: {
    sm: '0.85rem',
    md: '0.9rem',
    base: '1rem',
  },
  prose: '60ch',
} as const;

// d3-free: the scheme NAME is stored here; MatrixGrid resolves it to a d3 interpolator.
export const viz = {
  matrixScheme: 'Blues',
  cellStroke: color.border.strong,
  highlightStroke: '#e60',
  matrixText: '#222',
} as const;
