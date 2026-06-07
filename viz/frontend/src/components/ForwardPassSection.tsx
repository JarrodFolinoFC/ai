import { Flex } from 'antd';

import { color } from '../theme';
import { ForwardPassTable } from './ForwardPassTable';

interface Pass {
  logits: number[];
  exps: number[];
  expSum: number;
  probs: number[];
}

interface ForwardPassSectionProps {
  vocab: readonly string[];
  targetIdx: number;
  prevToken: string;
  // When flashed, the pre-step pass (which drove the gradient) is shown above
  // the post-step pass; otherwise only the current pass is shown.
  flashing: boolean;
  before: Pass;
  current: Pass;
}

const captionStyle = {
  fontSize: '0.8rem',
  fontWeight: 'bold' as const,
  color: color.text.secondary,
};

// The forward-pass tables for the worked example: the before/after pair while a
// single step is flashed, or just the current pass otherwise. Each caption sits
// directly above its table.
export function ForwardPassSection({
  vocab,
  targetIdx,
  prevToken,
  flashing,
  before,
  current,
}: ForwardPassSectionProps) {
  if (!flashing) {
    return <ForwardPassTable vocab={vocab} {...current} targetIdx={targetIdx} prevToken={prevToken} />;
  }
  return (
    <>
      <Flex vertical>
        <div style={captionStyle}>before this step — drives the gradient</div>
        <ForwardPassTable vocab={vocab} {...before} targetIdx={targetIdx} prevToken={prevToken} />
      </Flex>
      <Flex vertical>
        <div style={captionStyle}>after this step — result of the update</div>
        <ForwardPassTable vocab={vocab} {...current} targetIdx={targetIdx} prevToken={prevToken} />
      </Flex>
    </>
  );
}
