import React, { lazy } from 'react';
import { Navigate } from 'react-router-dom';
import Layout from './app/Layout';

/**
 * Route-based code splitting configuration.
 * Pages are lazily loaded via React.lazy() to optimize the initial bundle size and load time.
 */

const Dashboard = lazy(() => import('./app/pages/Dashboard'));
const Jobs = lazy(() => import('./app/pages/Jobs'));
const Requests = lazy(() => import('./app/pages/Requests'));
const Collections = lazy(() => import('./app/pages/Collections'));
const ProcessSKUs = lazy(() => import('./app/pages/ProcessSKUs'));
const Interactive = lazy(() => import('./app/pages/Interactive'));
const SkuResults = lazy(() => import('./app/pages/SkuResults'));
const RulesEngine = lazy(() => import('./app/pages/RulesEngine'));
const CatalogSearch = lazy(() => import('./app/pages/CatalogSearch'));
const Logs = lazy(() => import('./app/pages/Logs'));

const routes = () => [
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Navigate to="/dashboard" replace /> },
      { path: 'dashboard', element: <Dashboard /> },
      { path: 'jobs', element: <Jobs /> },
      { path: 'jobs/:jobId', element: <Jobs /> },
      { path: 'requests', element: <Requests /> },
      { path: 'collections', element: <Collections /> },
      { path: 'process-skus', element: <ProcessSKUs /> },
      { path: 'interactive', element: <Interactive /> },
      { path: 'sku-results', element: <SkuResults /> },
      { path: 'rules', element: <RulesEngine /> },
      { path: 'catalog', element: <CatalogSearch /> },
      { path: 'logs', element: <Logs /> },
      { path: '*', element: <Navigate to="/dashboard" replace /> },
    ],
  },
];

export default routes;
