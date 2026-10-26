import { useQuery } from '@tanstack/react-query';
import { getMe } from '../api';
import { useStore } from '../store';

// The server-confirmed identity for the active session (GET /auth/me). The backend enforces
// RBAC; the UI only mirrors it so users see locked controls instead of 403 errors.
export function useAuth() {
  const { authToken, apiKey } = useStore();
  const { data: me, isLoading } = useQuery({
    queryKey: ['auth-me', authToken, apiKey],
    queryFn: getMe,
    retry: false,
    staleTime: 60 * 1000,
  });
  const signedIn = Boolean(authToken || apiKey);
  // With AUTH_ENABLED=false the server reports anonymous visitors as admin; only a real login grants a role
  const role = signedIn ? me?.role : undefined;
  return {
    me,
    role,
    loading: signedIn && isLoading,
    signedIn,
    isAdmin: role === 'admin',
    canEditRules: role === 'admin' || role === 'rule_engineer',
  };
}
