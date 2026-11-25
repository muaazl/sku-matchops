import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const SRC_ROOT = path.dirname(fileURLToPath(import.meta.url));

const APPROVED_SRC_DIRS = ['components', 'hooks', 'pages', 'services', 'theme', 'utils'];
const APPROVED_SRC_FILES = ['App.jsx', 'architectureBoundaries.test.js', 'index.css', 'index.jsx', 'routes.jsx'];
const FORBIDDEN_LEGACY_ROOT_DIRS = ['app', 'lib', 'context'];
const FORBIDDEN_LEGACY_COMPONENT_DIRS = ['Common', 'Sidebar'];

function walkFiles(dir) {
  const results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkFiles(fullPath));
    } else if (entry.isFile()) {
      results.push(fullPath);
    }
  }
  return results;
}

function isTestFile(filename) {
  return /\.test\.(js|jsx|ts|tsx)$/.test(filename);
}

function extractImportSpecifiers(sourceText) {
  const specs = [];
  const staticImportRe = /(?:import|export)\s+(?:[\s\S]*?\s+from\s+)?['"]([^'"]+)['"]/g;
  const dynamicImportRe = /import\s*\(\s*['"]([^'"]+)['"]\s*\)/g;
  for (const re of [staticImportRe, dynamicImportRe]) {
    let match;
    while ((match = re.exec(sourceText)) !== null) {
      specs.push(match[1]);
    }
  }
  return specs;
}

function resolveLayer(relPosix) {
  const parts = relPosix.split('/');
  return parts.length === 1 ? 'root' : parts[0];
}

describe('Frontend architecture boundaries and naming', () => {
  it('contains only approved top-level directories and files in src/', () => {
    const entries = fs.readdirSync(SRC_ROOT, { withFileTypes: true });
    const dirs = entries
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort();
    const files = entries
      .filter((e) => e.isFile())
      .map((e) => e.name)
      .sort();

    for (const legacyDir of FORBIDDEN_LEGACY_ROOT_DIRS) {
      expect(dirs).not.toContain(legacyDir);
    }

    const componentDirs = fs
      .readdirSync(path.join(SRC_ROOT, 'components'), { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name);
    for (const legacyCompDir of FORBIDDEN_LEGACY_COMPONENT_DIRS) {
      expect(componentDirs).not.toContain(legacyCompDir);
    }

    expect(dirs).toEqual(APPROVED_SRC_DIRS);
    expect(files).toEqual(APPROVED_SRC_FILES);
  });

  it('enforces layer file naming conventions', () => {
    const allFiles = walkFiles(SRC_ROOT);
    const namingErrors = [];

    for (const absPath of allFiles) {
      const rel = path.relative(SRC_ROOT, absPath).split(path.sep).join('/');
      const layer = resolveLayer(rel);
      const basename = path.basename(absPath);
      if (isTestFile(basename)) continue;

      if (layer === 'pages' || layer === 'components') {
        if (!/^[A-Z][A-Za-z0-9]*\.jsx$/.test(basename)) {
          namingErrors.push(`${rel} must be PascalCase.jsx`);
        }
      } else if (layer === 'hooks') {
        if (!/^use[A-Z][A-Za-z0-9]*\.js$/.test(basename)) {
          namingErrors.push(`${rel} must match use*.js (camelCase)`);
        }
      } else if (layer === 'services' || layer === 'utils') {
        if (!/^[a-z][A-Za-z0-9]*\.js$/.test(basename)) {
          namingErrors.push(`${rel} must be camelCase.js`);
        }
      }
    }

    expect(namingErrors).toEqual([]);
  });

  it('enforces layer dependency rules and prohibits raw fetch/axios in UI layers', () => {
    const allFiles = walkFiles(SRC_ROOT);
    const violations = [];

    for (const absPath of allFiles) {
      const rel = path.relative(SRC_ROOT, absPath).split(path.sep).join('/');
      const layer = resolveLayer(rel);
      const basename = path.basename(absPath);
      if (isTestFile(basename) || layer === 'theme' || rel === 'index.css') continue;

      const content = fs.readFileSync(absPath, 'utf-8');
      const specifiers = extractImportSpecifiers(content);

      for (const spec of specifiers) {
        let targetLayer = null;
        if (spec.startsWith('.')) {
          const resolved = path.resolve(path.dirname(absPath), spec);
          const resolvedRel = path.relative(SRC_ROOT, resolved).split(path.sep).join('/');
          targetLayer = resolveLayer(resolvedRel);
        }

        if (layer === 'pages' || layer === 'components') {
          if (targetLayer === 'services') {
            violations.push(`${rel} imports service layer '${spec}' directly (use hooks/)`);
          }
          if (spec === 'axios') {
            violations.push(`${rel} imports axios directly`);
          }
        }

        if (layer === 'hooks') {
          if (targetLayer === 'pages' || targetLayer === 'components') {
            violations.push(`${rel} imports upward UI layer '${spec}'`);
          }
          if (spec === 'axios') {
            violations.push(`${rel} imports axios directly`);
          }
        }

        if (layer === 'services') {
          if (targetLayer === 'pages' || targetLayer === 'components' || targetLayer === 'hooks') {
            violations.push(`${rel} imports forbidden upward layer '${spec}'`);
          }
          if (spec === 'react' || spec === 'react-dom') {
            violations.push(`${rel} imports React inside services/`);
          }
        }

        if (layer === 'utils') {
          if (['pages', 'components', 'hooks', 'services'].includes(targetLayer)) {
            violations.push(`${rel} imports non-utility layer '${spec}'`);
          }
          if (spec === 'axios' || spec === 'react' || spec === 'react-dom') {
            violations.push(`${rel} imports forbidden runtime '${spec}' inside utils/`);
          }
        }
      }

      if (layer !== 'services' && /\bfetch\s*\(/.test(content)) {
        violations.push(`${rel} calls fetch() outside services/`);
      }
    }

    expect(violations).toEqual([]);
  });
});
