import { useState } from 'react';
import { Button } from 'antd';

import { color, font, radius, space } from '../theme';
import { ConsoleOutput } from './ConsoleOutput';

interface JsRunnerProps {
  // Code to seed the editor with; the user can edit before running.
  initialCode: string;
}

// Format a console.log argument roughly the way the browser console would.
function format(value: unknown): string {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

// An editable JavaScript scratchpad: edit the code, Run it in a sandboxed
// Function with console.log captured, and see the output in ConsoleOutput.
export function JsRunner({ initialCode }: JsRunnerProps) {
  const [code, setCode] = useState(initialCode);
  const [lines, setLines] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    const logs: string[] = [];
    const sandboxConsole = {
      log: (...args: unknown[]) => logs.push(args.map(format).join(' ')),
    };
    try {
      // The code is generated locally (not user-supplied content), and runs
      // with a sandboxed console — no access to the real console or DOM refs.
      // eslint-disable-next-line @typescript-eslint/no-implied-eval
      new Function('console', code)(sandboxConsole);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? `${e.name}: ${e.message}` : String(e));
    }
    setLines(logs);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: space.md }}>
      <textarea
        value={code}
        onChange={(e) => setCode(e.target.value)}
        spellCheck={false}
        style={{
          fontFamily: 'monospace',
          fontSize: font.size.sm,
          padding: space.md,
          borderRadius: radius.sm,
          border: `1px solid ${color.border.default}`,
          background: color.bg.subtle,
          minHeight: '40vh',
          resize: 'vertical',
          width: '100%',
          boxSizing: 'border-box',
        }}
      />
      <div>
        <Button type="primary" onClick={run}>
          Run ▶
        </Button>
      </div>
      <ConsoleOutput lines={lines} error={error} />
    </div>
  );
}
