import { font, radius, space } from '../theme';

interface ConsoleOutputProps {
  // Captured console.log lines, in order.
  lines: string[];
  // A thrown error, if the run failed.
  error: string | null;
}

// A terminal-style panel that renders captured console.log output (and any
// runtime error) from a JsRunner.
export function ConsoleOutput({ lines, error }: ConsoleOutputProps) {
  return (
    <div
      style={{
        fontFamily: 'monospace',
        fontSize: font.size.sm,
        background: '#1e1e1e',
        color: '#e0e0e0',
        padding: space.md,
        borderRadius: radius.sm,
        minHeight: '8rem',
        overflow: 'auto',
        whiteSpace: 'pre-wrap',
      }}
    >
      {lines.length === 0 && !error && (
        <span style={{ color: '#888' }}>// console output appears here</span>
      )}
      {lines.map((line, i) => (
        <div key={i}>{line}</div>
      ))}
      {error && <div style={{ color: '#ff6b6b' }}>{error}</div>}
    </div>
  );
}
