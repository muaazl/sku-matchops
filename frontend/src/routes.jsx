import React, { lazy } from 'react';
import { Navigate } from 'react-router-dom';
import Layout from './app/Layout';
import RequireLogin from './app/components/RequireLogin';

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
const Guide = lazy(() => import('./app/pages/Guide'));
const QueueDlq = lazy(() => import('./app/pages/QueueDlq'));

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
      // Logged-in only: Collections, Process SKUs, Rules Engine; Queue & DLQ is admin-only
      {
        path: 'collections',
        element: (
          <RequireLogin>
            <Collections />
          </RequireLogin>
        ),
      },
      {
        path: 'process-skus',
        element: (
          <RequireLogin>
            <ProcessSKUs />
          </RequireLogin>
        ),
      },
      { path: 'interactive', element: <Interactive /> },
      { path: 'sku-results', element: <SkuResults /> },
      {
        path: 'rules',
        element: (
          <RequireLogin>
            <RulesEngine />
          </RequireLogin>
        ),
      },
      { path: 'catalog', element: <CatalogSearch /> },
      { path: 'logs', element: <Logs /> },
      {
        path: 'queue',
        element: (
          <RequireLogin admin>
            <QueueDlq />
          </RequireLogin>
        ),
      },
      { path: 'guide', element: <Guide /> },
      { path: '*', element: <Navigate to="/dashboard" replace /> },
    ],
  },
];

export default routes;
