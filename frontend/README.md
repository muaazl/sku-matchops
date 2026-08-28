# SKU MatchOps — Frontend

React + Vite + MUI console for SKU MatchOps. Built on the Qdrant Web UI design
system (theme, layout, components) and trimmed down to just this app.

## Run

```bash
npm install -g pnpm  # If pnpm is not already installed
pnpm install
pnpm start           # http://localhost:5173
```

## Structure

```
src/
  routes.jsx             # route table
  app/                   # the entire app
    Layout.jsx           # app shell (AppBar + sidebar)
    SkuSidebar.jsx       # left navigation
    pages/               # the 11 pages (Dashboard, Jobs, Requests, ...)
    components/          # shared UI (ui.jsx, Charts.jsx, JsonBlock.jsx)
    data/mock.js         # dummy data — swap this out for real API calls
    utils.js
  theme/                 # MUI theme (dark / light / high-contrast)
  components/, context/, lib/   # shared pieces the app imports (Logo, toggles, color-context, ...)
```

## Connecting the real API

All pages currently read from `src/app/data/mock.js`. To go live, replace the
`data/mock` imports in each page with `fetch`/axios calls to your FastAPI backend
(e.g. `GET /jobs`, `GET /api-requests`, `GET /processed-skus`, ...). A lightweight
dummy backend that serves those endpoints lives in
`../sku-matchops/backend/dev_server.py`.
