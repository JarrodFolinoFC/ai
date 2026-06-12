import { FormulaDisplay } from '../components/FormulaDisplay';
import { FORMULAS, FORMULA_SECTIONS } from '../formulas';
import { color, space, radius, font } from '../theme';

// Read-only catalog of every static formula in the registry. To change a formula,
// edit its entry in src/formulas.ts — it updates here and on every page that
// references it.
export function FormulasPage() {
  return (
    <div>
      <h2>Formulas</h2>
      <p style={{ maxWidth: font.prose, color: color.text.secondary }}>
        Every static formula used across the visualizer, sourced from{' '}
        <code>src/formulas.ts</code>. Edit a formula there to change it everywhere
        it is referenced.
      </p>

      {FORMULA_SECTIONS.map((section) => (
        <section key={section.title} style={{ marginBottom: space.xl }}>
          <h3>{section.title}</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
            {section.keys.map((key) => (
              <div
                key={key}
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'minmax(12rem, auto) 1fr',
                  gap: space.lg,
                  alignItems: 'center',
                  padding: space.sm,
                  border: `1px solid ${color.border.default}`,
                  borderRadius: radius.sm,
                  background: color.bg.surface,
                }}
              >
                <code style={{ fontSize: font.size.sm, color: color.text.secondary }}>
                  {key}
                </code>
                <FormulaDisplay latex={FORMULAS[key]} />
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
