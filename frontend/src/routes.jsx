import React, { lazy } from 'react';
import { Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Jobs = lazy(() => import('./pages/Jobs'));
const Requests = lazy(() => import('./pages/Requests'));
const Collections = lazy(() => import('./pages/Collections'));
const ProcessSKUs = lazy(() => import('./pages/ProcessSKUs'));
const Interactive = lazy(() => import('./pages/Interactive'));
const SkuResults = lazy(() => import('./pages/SkuResults'));
const RulesEngine = lazy(() => import('./pages/RulesEngine'));
const CatalogSearch = lazy(() => import('./pages/CatalogSearch'));
const Logs = lazy(() => import('./pages/Logs'));
const Guide = lazy(() => import('./pages/Guide'));

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
      { path: 'guide', element: <Guide /> },
      { path: '*', element: <Navigate to="/dashboard" replace /> },
    ],
  },
];

export default routes;
