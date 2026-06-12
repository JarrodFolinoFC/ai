import { Alert } from 'antd';

import { space } from '../theme';

interface Props {
  message: string | null;
}

export function ErrorBanner({ message }: Props) {
  if (!message) return null;
  return (
    <Alert
      type="error"
      message={message}
      style={{ margin: `${space.sm} 0` }}
    />
  );
}
