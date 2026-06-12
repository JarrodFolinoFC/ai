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
  const maxAge = 60 * 60 * 24 * 365;
  document.cookie = `${NAV_COOKIE}=${open ? '1' : '0'}; max-age=${maxAge}; path=/; SameSite=Lax`;
}

import { Stage1FlowPage } from './pages/Stage1FlowPage';

interface NavGroup {
  heading: string;
  collapsible?: boolean;
  items: { path: string; label: string }[];
}

const NAV: NavGroup[] = [
  {
    heading: 'Bigram',
    items: [
      { path: '/bigram/training', label: 'Training' },
    ],
  }
];

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
          <Route path="/bigram/training" element={<Stage1FlowPage />} />
        </Routes>
      </Layout.Content>
    </Layout>
  );
}
