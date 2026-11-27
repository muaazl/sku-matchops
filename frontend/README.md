# SKU MatchOps — Frontend

React 18 + Vite + MUI console for SKU MatchOps.

## Run

```bash
npm install -g pnpm  # If pnpm is not already installed
pnpm install
pnpm start           # http://localhost:5173
```

## Structure

```text
src/
  index.jsx              # Application bootstrap & React Query / Snackbar providers
  App.jsx                # Root app shell (Theme mode, ServerOfflineModal, Router)
  routes.jsx             # Lazy-loaded route configuration
  pages/                 # PascalCase.jsx route views
  components/            # PascalCase.jsx reusable UI, layout, chart, and feature components
  hooks/                 # use*.js React hooks, Query/Mutation orchestrators & Zustand store
  services/              # camelCase.js Axios/SSE clients & backend API services
  utils/                 # camelCase.js pure helpers, formatters, parsers & constants
  theme/                 # MUI theme palettes (dark / light / high-contrast)
```

## Checks

```bash
pnpm run lint
pnpm run format:check
pnpm run test:run
pnpm run build
pnpm run test:e2e
```
