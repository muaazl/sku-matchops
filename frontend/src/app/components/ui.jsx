import React from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Dialog,
  Drawer,
  IconButton,
  Skeleton,
  Stack,
  Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { WarningAmber, HelpOutline, Inbox, Close } from '@mui/icons-material';

// Map a job/request status to an MUI chip color.
const STATUS_COLOR = {
  completed: 'success',
  done: 'success',
  running: 'info',
  queued: 'default',
  pending: 'warning',
  failed: 'error',
  cancelled: 'default',
  approved: 'success',
  rejected: 'error',
  matched: 'success',
  review: 'warning',
  confident: 'success',
};

export function StatusChip({ status, ...props }) {
  const color = STATUS_COLOR[status] ?? 'default';
  return (
    <Chip
      size="small"
      label={status}
      color={color}
      variant={color === 'default' ? 'outlined' : 'filled'}
      sx={{ textTransform: 'capitalize', fontWeight: 500, ...(props.sx || {}) }}
      {...props}
    />
  );
}

StatusChip.propTypes = {
  status: PropTypes.string.isRequired,
  sx: PropTypes.object,
};

export function HttpStatusChip({ code }) {
  const color = code >= 500 ? 'error' : code >= 400 ? 'warning' : code >= 200 && code < 300 ? 'success' : 'default';
  return <Chip size="small" label={code} color={color} sx={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }} />;
}

HttpStatusChip.propTypes = { code: PropTypes.number.isRequired };

export function PageHeader({ title, subtitle, actions }) {
  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 2,
        mb: 3,
      }}
    >
      <Box>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {subtitle}
          </Typography>
        )}
      </Box>
      {actions && <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>{actions}</Box>}
    </Box>
  );
}

PageHeader.propTypes = {
  title: PropTypes.node.isRequired,
  subtitle: PropTypes.node,
  actions: PropTypes.node,
};

// Consistent page container that leaves room under the fixed AppBar.
export function PageContainer({ children }) {
  return <Box sx={{ p: { xs: 2, md: 4 }, maxWidth: 1400, mx: 'auto' }}>{children}</Box>;
}

PageContainer.propTypes = { children: PropTypes.node };

export function SideDrawer({ open, onClose, title, children, width = 480 }) {
  return (
    <Drawer anchor="right" open={open} onClose={onClose} PaperProps={{ sx: { width: { xs: '100%', sm: width } } }}>
      {open && (
        <Box sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
              {title}
            </Typography>
            <IconButton onClick={onClose} size="small" aria-label="Close drawer">
              <Close sx={{ fontSize: 20 }} />
            </IconButton>
          </Box>
          <Box sx={{ flexGrow: 1, overflowY: 'auto' }}>{children}</Box>
        </Box>
      )}
    </Drawer>
  );
}

SideDrawer.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  title: PropTypes.node.isRequired,
  children: PropTypes.node,
  width: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
};

// Shared visual language for "simple" modals across the app — an icon badge, a centered
// title/description, optional extra content, and a row of actions. Modeled after
// ServerOfflineModal's look (icon-in-circle, rounded bordered card, centered text) so
// every confirmation/prompt dialog in the app reads as the same component family, while
// still using MUI's Dialog underneath for proper focus-trap/ESC/backdrop-click behavior.
export function IconModal({
  open,
  onClose,
  icon: Icon,
  iconColor,
  title,
  description,
  children,
  actions,
  maxWidth = 420,
  hideCloseButton = false,
}) {
  const theme = useTheme();
  return (
    <Dialog
      open={open}
      onClose={onClose}
      slotProps={{
        backdrop: {
          sx: {
            backdropFilter: 'blur(8px)',
            backgroundColor: theme.palette.mode === 'dark' ? 'rgba(0, 0, 0, 0.6)' : 'rgba(0, 0, 0, 0.3)',
          },
        },
        paper: {
          sx: {
            maxWidth,
            width: '100%',
            borderRadius: 2,
            border: `1px solid ${theme.palette.divider}`,
            p: 4,
            position: 'relative',
          },
        },
      }}
    >
      {!hideCloseButton && onClose && (
        <IconButton
          onClick={onClose}
          size="small"
          aria-label="Close dialog"
          sx={{ position: 'absolute', top: 12, right: 12, color: 'text.secondary' }}
        >
          <Close sx={{ fontSize: 18 }} />
        </IconButton>
      )}

      <Box sx={{ textAlign: 'center' }}>
        {Icon && (
          <Box
            sx={{
              display: 'inline-flex',
              p: 1.5,
              borderRadius: '50%',
              backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
              mb: 2,
            }}
          >
            {React.isValidElement(Icon) ? (
              Icon
            ) : (
              <Icon sx={{ fontSize: 32, color: iconColor || theme.palette.text.secondary }} />
            )}
          </Box>
        )}

        <Typography variant="h6" sx={{ fontWeight: 600, mb: description ? 1 : 0 }}>
          {title}
        </Typography>

        {description && (
          <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.5 }}>
            {description}
          </Typography>
        )}
      </Box>

      {children && <Box sx={{ textAlign: 'left', mt: 3 }}>{children}</Box>}

      {actions && (
        <Stack direction="row" spacing={1.5} sx={{ mt: 3 }}>
          {actions}
        </Stack>
      )}
    </Dialog>
  );
}

IconModal.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func,
  icon: PropTypes.elementType,
  iconColor: PropTypes.string,
  title: PropTypes.node.isRequired,
  description: PropTypes.node,
  children: PropTypes.node,
  actions: PropTypes.node,
  maxWidth: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  hideCloseButton: PropTypes.bool,
};

export function ConfirmDialog({
  open,
  onClose,
  title,
  message,
  onConfirm,
  confirmText = 'Yes',
  cancelText = 'No',
  confirmColor = 'primary',
}) {
  const theme = useTheme();
  const isDestructive = confirmColor === 'error';
  return (
    <IconModal
      open={open}
      onClose={onClose}
      icon={isDestructive ? WarningAmber : HelpOutline}
      iconColor={isDestructive ? theme.palette.error.main : undefined}
      title={title}
      description={message}
      actions={
        <>
          <Button onClick={onClose} color="inherit" variant="outlined" fullWidth>
            {cancelText}
          </Button>
          <Button variant="contained" color={confirmColor} onClick={onConfirm} autoFocus fullWidth>
            {confirmText}
          </Button>
        </>
      }
    />
  );
}

ConfirmDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  title: PropTypes.string.isRequired,
  message: PropTypes.node.isRequired,
  onConfirm: PropTypes.func.isRequired,
  confirmText: PropTypes.string,
  cancelText: PropTypes.string,
  confirmColor: PropTypes.string,
};

// Standardized KPI / Metric Card with built-in loading skeleton,
// accessible markup, and smooth micro-interaction hover state.
export function StatCard({
  icon: Icon,
  label,
  value,
  subtext,
  color = 'primary',
  isLoading = false,
  trend,
  onClick,
  sx = {},
}) {
  const theme = useTheme();
  const paletteColor = theme.palette[color] || theme.palette.primary;

  return (
    <Card
      role="region"
      aria-label={typeof label === 'string' ? label : 'Metric card'}
      onClick={onClick}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        cursor: onClick ? 'pointer' : 'default',
        borderRadius: 2,
        border: `1px solid ${theme.palette.divider}`,
        transition:
          'transform 180ms cubic-bezier(0.4, 0, 0.2, 1), box-shadow 180ms cubic-bezier(0.4, 0, 0.2, 1), border-color 180ms ease',
        '&:hover': onClick
          ? {
              transform: 'translateY(-2px)',
              boxShadow: theme.palette.mode === 'dark' ? '0 6px 20px rgba(0,0,0,0.45)' : '0 6px 20px rgba(0,0,0,0.08)',
              borderColor: paletteColor.main,
            }
          : {
              borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.2)',
            },
        ...sx,
      }}
    >
      <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2.5, '&:last-child': { pb: 2.5 } }}>
        {Icon && (
          <Box
            sx={{
              width: 48,
              height: 48,
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: paletteColor.main + '1f',
              color: paletteColor.main,
              flexShrink: 0,
            }}
          >
            {React.isValidElement(Icon) ? Icon : <Icon sx={{ fontSize: 24 }} />}
          </Box>
        )}
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          {isLoading ? (
            <>
              <Skeleton variant="text" width="60%" height={32} sx={{ mb: 0.5 }} />
              <Skeleton variant="text" width="85%" height={20} />
              {subtext && <Skeleton variant="text" width="45%" height={16} sx={{ mt: 0.5 }} />}
            </>
          ) : (
            <>
              <Typography variant="h5" sx={{ fontWeight: 700, lineHeight: 1.15, letterSpacing: '-0.02em' }}>
                {value}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, fontWeight: 500 }} noWrap>
                {label}
              </Typography>
              {(subtext || trend) && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5, flexWrap: 'wrap' }}>
                  {trend && (
                    <Chip
                      size="small"
                      label={trend.label}
                      color={trend.direction === 'up' ? 'success' : trend.direction === 'down' ? 'error' : 'default'}
                      sx={{ height: 20, fontSize: '0.7rem', fontWeight: 600 }}
                    />
                  )}
                  {subtext && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      {subtext}
                    </Typography>
                  )}
                </Box>
              )}
            </>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}

StatCard.propTypes = {
  icon: PropTypes.elementType,
  label: PropTypes.node.isRequired,
  value: PropTypes.node,
  subtext: PropTypes.node,
  color: PropTypes.string,
  isLoading: PropTypes.bool,
  trend: PropTypes.shape({
    label: PropTypes.string,
    direction: PropTypes.oneOf(['up', 'down', 'neutral']),
  }),
  onClick: PropTypes.func,
  sx: PropTypes.object,
};

// Standardized Section & Chart container with consistent border tokens,
// headers, actions, and built-in skeleton loader.
export function SectionCard({
  title,
  subheader,
  action,
  children,
  isLoading = false,
  minHeight,
  sx = {},
  contentSx = {},
  ...props
}) {
  const theme = useTheme();

  return (
    <Card
      sx={{
        height: '100%',
        borderRadius: 2,
        border: `1px solid ${theme.palette.divider}`,
        display: 'flex',
        flexDirection: 'column',
        ...sx,
      }}
      {...props}
    >
      {(title || action) && (
        <CardHeader
          title={title}
          subheader={subheader}
          action={action}
          titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
          subheaderTypographyProps={{ variant: 'body2', color: 'text.secondary' }}
          sx={{ pb: 1, pt: 2, px: 2.5 }}
        />
      )}
      <CardContent
        sx={{
          flexGrow: 1,
          p: 2.5,
          minHeight,
          '&:last-child': { pb: 2.5 },
          ...contentSx,
        }}
      >
        {isLoading ? (
          <Box
            sx={{
              height: minHeight || 260,
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
              justifyContent: 'center',
            }}
          >
            <Skeleton variant="rounded" width="100%" height="80%" sx={{ borderRadius: 2 }} />
            <Skeleton variant="text" width="40%" height={24} />
          </Box>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
}

SectionCard.propTypes = {
  title: PropTypes.node,
  subheader: PropTypes.node,
  action: PropTypes.node,
  children: PropTypes.node,
  isLoading: PropTypes.bool,
  minHeight: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  sx: PropTypes.object,
  contentSx: PropTypes.object,
};

// Standardized Empty State with icon badge, clear heading,
// optional explanatory copy, and action button.
export function EmptyState({ icon: Icon = Inbox, title, description, action, minHeight = 220, sx = {} }) {
  const theme = useTheme();

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        p: 3,
        minHeight,
        ...sx,
      }}
    >
      <Box
        sx={{
          width: 52,
          height: 52,
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.04)',
          color: 'text.secondary',
          mb: 2,
        }}
      >
        {React.isValidElement(Icon) ? Icon : <Icon sx={{ fontSize: 26 }} />}
      </Box>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: description ? 0.75 : 0 }}>
        {title}
      </Typography>
      {description && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ maxWidth: 440, mb: action ? 2.5 : 0, lineHeight: 1.5 }}
        >
          {description}
        </Typography>
      )}
      {action && <Box sx={{ mt: description ? 0 : 2 }}>{action}</Box>}
    </Box>
  );
}

EmptyState.propTypes = {
  icon: PropTypes.elementType,
  title: PropTypes.node.isRequired,
  description: PropTypes.node,
  action: PropTypes.node,
  minHeight: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  sx: PropTypes.object,
};

// Layout-preserving table skeleton loader.
export function TableSkeleton({ rows = 5, height = 44, sx = {} }) {
  return (
    <Box sx={{ width: '100%', overflow: 'hidden', ...sx }}>
      <Stack spacing={1.5}>
        <Skeleton variant="rounded" height={40} sx={{ borderRadius: 1 }} />
        {Array.from({ length: rows }).map((_, idx) => (
          <Skeleton key={idx} variant="rounded" height={height} sx={{ borderRadius: 1, opacity: 1 - idx * 0.12 }} />
        ))}
      </Stack>
    </Box>
  );
}

TableSkeleton.propTypes = {
  rows: PropTypes.number,
  height: PropTypes.number,
  sx: PropTypes.object,
};

// Reusable card skeleton placeholder.
export function CardSkeleton({ height = 120, sx = {} }) {
  return <Skeleton variant="rounded" height={height} sx={{ borderRadius: 2, ...sx }} />;
}

CardSkeleton.propTypes = {
  height: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  sx: PropTypes.object,
};
