import React from 'react';
import PropTypes from 'prop-types';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

// Route guard for pages hidden from logged-out visitors (typing the URL redirects to the
// dashboard). `admin` additionally requires the admin role.
export default function RequireLogin({ admin = false, children }) {
  const { signedIn, isAdmin, loading } = useAuth();
  if (!signedIn) return <Navigate to="/dashboard" replace />;
  if (admin) {
    if (loading) return null; // role not confirmed yet; don't bounce an admin on refresh
    if (!isAdmin) return <Navigate to="/dashboard" replace />;
  }
  return children;
}

RequireLogin.propTypes = {
  admin: PropTypes.bool,
  children: PropTypes.node.isRequired,
};
