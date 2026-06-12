import { Flex } from 'antd';

import { color, space, font, radius } from '../theme';
import { Panel } from './Panel';

interface Props {
  vocab: readonly string[];
}

// Shows the token -> id mapping for the vocabulary, e.g. the=0, cat=1, ...
// Each token's array index IS its id, so we just render them in order.
export function TokenIdMap({ vocab }: Props) {
  return (
    <Panel title={<>Token &rarr; ID</>} style={{ width: 'fit-content' }}>
      <Flex vertical align="start" gap={space.xs}>
        {vocab.map((token, id) => (
          <Flex
            key={token}
            align="center"
            gap={space.sm}
            style={{
              padding: `${space.xs} ${space.sm}`,
              border: `1px solid ${color.border.default}`,
              borderRadius: radius.sm,
              background: color.bg.subtle,
              fontSize: font.size.sm,
            }}
          >
            <span style={{ fontFamily: 'monospace', minWidth: '3.5ch' }}>{token}</span>
            <span style={{ color: color.text.muted }}>{id}</span>
          </Flex>
        ))}
      </Flex>
    </Panel>
  );
}
