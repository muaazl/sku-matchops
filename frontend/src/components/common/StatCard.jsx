import React from 'react';
import PropTypes from 'prop-types';
import { Box, Card, CardContent, Chip, Skeleton, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';

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
  const hoverStyles = onClick
    ? {
        transform: 'translateY(-2px)',
        boxShadow: theme.palette.mode === 'dark' ? '0 6px 20px rgba(0,0,0,0.45)' : '0 6px 20px rgba(0,0,0,0.08)',
        borderColor: paletteColor.main,
      }
    : {
        borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.2)',
      };

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
          'transform 180ms cubic-bezier(0.4, 0, 0.2, 1), ' +
          'box-shadow 180ms cubic-bezier(0.4, 0, 0.2, 1), border-color 180ms ease',
        '&:hover': hoverStyles,
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
  icon: PropTypes.oneOfType([PropTypes.elementType, PropTypes.node]),
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
