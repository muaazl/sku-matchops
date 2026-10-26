import { useQuery } from '@tanstack/react-query';
import { getMe } from '../api';
import { useStore } from '../store';

// The server-confirmed identity for the active session (GET /auth/me). The backend enforces
// RBAC; the UI only mirrors it so users see locked controls instead of 403 errors.
export function useAuth() {
  const { authToken, apiKey, tenantId } = useStore();
  const { data: me, isLoading } = useQuery({
    queryKey: ['auth-me', authToken, apiKey, tenantId],
    queryFn: getMe,
    retry: false,
    staleTime: 60 * 1000,
  });
  const role = me?.role;
  return {
    me,
    role,
    loading: isLoading,
    signedIn: Boolean(authToken || apiKey),
    isAdmin: role === 'admin',
    canEditRules: role === 'admin' || role === 'rule_engineer',
    isPlatformAdmin: Boolean(me?.platform_admin),
  };
}
