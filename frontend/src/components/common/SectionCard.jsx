import React from 'react';
import PropTypes from 'prop-types';
import { Box, Card, CardContent, CardHeader, Skeleton } from '@mui/material';
import { useTheme } from '@mui/material/styles';

export function SectionCard({
  icon: Icon,
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
  let avatar = null;
  if (Icon) {
    avatar = React.isValidElement(Icon) ? Icon : <Icon sx={{ fontSize: 20 }} />;
  }

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
      {(title || action || avatar) && (
        <CardHeader
          avatar={avatar}
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
  icon: PropTypes.oneOfType([PropTypes.elementType, PropTypes.node]),
  title: PropTypes.node,
  subheader: PropTypes.node,
  action: PropTypes.node,
  children: PropTypes.node,
  isLoading: PropTypes.bool,
  minHeight: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  sx: PropTypes.object,
  contentSx: PropTypes.object,
};
