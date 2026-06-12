import { useEffect, useState } from 'react';
import { Layout, Menu } from 'antd';
import { Route, Routes, useLocation, useNavigate } from 'react-router-dom';

import { color, space, font } from './theme';

const NAV_COOKIE = 'navOpen';

function readNavCookie(): boolean {
  const match = document.cookie.match(/(?:^|; )navOpen=([01])/);
  return match ? match[1] === '1' : true;
}

function writeNavCookie(open: boolean) {
  // 1-year cookie, root path, lax SameSite (no cross-site need)
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `${NAV_COOKIE}=${open ? '1' : '0'}; max-age=${maxAge}; path=/; SameSite=Lax`;
}

import { BigramBackwardPage } from './pages/BigramBackwardPage';
import { CombinedInputPage } from './pages/CombinedInputPage';
import { CrossEntropyLossPage } from './pages/CrossEntropyLossPage';
import { LowerTriMaskPage } from './pages/LowerTriMaskPage';
import { NormalizeRowsPage } from './pages/NormalizeRowsPage';
import { OneHotTruthPage } from './pages/OneHotTruthPage';
import { PositionEmbedPage } from './pages/PositionEmbedPage';
import { SgdStepPage } from './pages/SgdStepPage';
import { SoftmaxPage } from './pages/SoftmaxPage';
import { Stage1FlowPage } from './pages/Stage1FlowPage';
import { Stage2FlowPage } from './pages/Stage2FlowPage';
import { AttentionConceptsReferencePage } from './pages/AttentionConceptsReferencePage';
import { AttentionWeightedSumPage } from './pages/AttentionWeightedSumPage';
import { CausalScoreMaskPage } from './pages/CausalScoreMaskPage';
import { Stage2cIFlowPage } from './pages/Stage2cIFlowPage';
import { TokenEmbedPage } from './pages/TokenEmbedPage';
import { UnembedHeadPage } from './pages/UnembedHeadPage';
import { FormulasPage } from './pages/FormulasPage';

interface NavGroup {
  heading: string;
  // Collapsible groups render as an expand/collapse submenu; the rest render as
  // static section headings.
  collapsible?: boolean;
  items: { path: string; label: string }[];
}

const NAV: NavGroup[] = [
  {
    heading: 'Stage 1 (bigram)',
    items: [
      { path: '/stage1-flow', label: 'Stage 1 Flow' },
    ],
  },
  {
    heading: 'Stage 2a',
    items: [
      { path: '/stage2-flow', label: 'Stage 2a Flow' },
    ],
  },
  {
    heading: 'Stage 2c-i (hand-weighted attn)',
    items: [
      { path: '/attention-concepts', label: 'Attention Concepts (read me)' },
      { path: '/stage2c-i-flow', label: 'Stage 2c-i Flow' },
      { path: '/causal-score-mask', label: 'Causal Score Mask' },
      { path: '/attention-weighted-sum', label: 'Attention-Weighted Sum' },
    ],
  },
  {
    heading: 'Formulas',
    collapsible: true,
    items: [
      { path: '/formulas', label: 'Catalog' },
      { path: '/softmax', label: 'Softmax' },
      { path: '/one-hot-truth', label: 'One-Hot Truth' },
      { path: '/cross-entropy-loss', label: 'Cross-Entropy Loss' },
      { path: '/bigram-backward', label: 'Bigram Backward' },
      { path: '/sgd-step', label: 'SGD Step' },
      { path: '/token-embed', label: 'Token Embedding' },
      { path: '/position-embed', label: 'Position Embedding' },
      { path: '/combined-input', label: 'Combined Input' },
      { path: '/lower-tri-mask', label: 'Lower-Tri Mask' },
      { path: '/normalize-rows', label: 'Normalize Rows' },
      { path: '/unembed-head', label: 'Unembed Head' },
    ],
  },
];

// The collapsible group (if any) that contains a given route.
function groupForPath(path: string): string | undefined {
  return NAV.find((g) => g.collapsible && g.items.some((i) => i.path === path))
    ?.heading;
}

export default function App() {
  const [navOpen, setNavOpen] = useState<boolean>(() => readNavCookie());
  const location = useLocation();
  const navigate = useNavigate();
  useEffect(() => {
    writeNavCookie(navOpen);
  }, [navOpen]);

  // Keep the collapsible group open whenever the active route lives inside it
  // (e.g. landing on /softmax directly), while still letting the user toggle it.
  const [openKeys, setOpenKeys] = useState<string[]>(() => {
    const g = groupForPath(location.pathname);
    return g ? [g] : [];
  });
  useEffect(() => {
    const g = groupForPath(location.pathname);
    if (g) setOpenKeys((keys) => (keys.includes(g) ? keys : [...keys, g]));
  }, [location.pathname]);

  const menuItems = NAV.map((group) => {
    const children = group.items.map((n) => ({ key: n.path, label: n.label }));
    return group.collapsible
      ? { key: group.heading, label: group.heading, children }
      : { key: group.heading, label: group.heading, type: 'group' as const, children };
  });

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout.Sider
        width={240}
        collapsible
        collapsed={!navOpen}
        onCollapse={(collapsed) => setNavOpen(!collapsed)}
        collapsedWidth={48}
        theme="light"
        style={{ borderRight: `1px solid ${color.border.default}` }}
      >
        {navOpen && (
          <div style={{ padding: `${space.md} ${space.lg}` }}>
            <h1 style={{ fontSize: font.size.base, margin: 0, whiteSpace: 'nowrap' }}>
              llm2 visualizer
            </h1>
          </div>
        )}
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          openKeys={openKeys}
          onOpenChange={(keys) => setOpenKeys(keys as string[])}
          onClick={({ key }) => navigate(key)}
          items={menuItems}
          style={{ borderInlineEnd: 0 }}
        />
      </Layout.Sider>
      <Layout.Content style={{ padding: space.lg }}>
        <Routes>
          <Route path="/" element={<SoftmaxPage />} />
          <Route path="/stage1-flow" element={<Stage1FlowPage />} />
          <Route path="/stage2-flow" element={<Stage2FlowPage />} />
          <Route path="/attention-concepts" element={<AttentionConceptsReferencePage />} />
          <Route path="/stage2c-i-flow" element={<Stage2cIFlowPage />} />
          <Route path="/causal-score-mask" element={<CausalScoreMaskPage />} />
          <Route path="/attention-weighted-sum" element={<AttentionWeightedSumPage />} />
          <Route path="/softmax" element={<SoftmaxPage />} />
          <Route path="/one-hot-truth" element={<OneHotTruthPage />} />
          <Route path="/cross-entropy-loss" element={<CrossEntropyLossPage />} />
          <Route path="/bigram-backward" element={<BigramBackwardPage />} />
          <Route path="/sgd-step" element={<SgdStepPage />} />
          <Route path="/token-embed" element={<TokenEmbedPage />} />
          <Route path="/position-embed" element={<PositionEmbedPage />} />
          <Route path="/combined-input" element={<CombinedInputPage />} />
          <Route path="/lower-tri-mask" element={<LowerTriMaskPage />} />
          <Route path="/normalize-rows" element={<NormalizeRowsPage />} />
          <Route path="/unembed-head" element={<UnembedHeadPage />} />
          <Route path="/formulas" element={<FormulasPage />} />
        </Routes>
      </Layout.Content>
    </Layout>
  );
}
