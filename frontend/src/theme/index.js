import { createTheme as createMuiTheme } from '@mui/material/styles';
import { alpha } from '@mui/material/styles';
import { deepmerge } from '@mui/utils';
import { darkThemeOptions } from './dark-theme';
import { lightThemeOptions } from './light-theme';
import { highContrastThemeOptions } from './high-contrast-theme';

const themeOptions = {
  typography: {
    // Portal design token (src/index.css)
    fontFamily: 'var(--portal-font-sans)',
    allVariants: {
      fontFeatureSettings: "'ss01' on, 'ss05' on, 'ss06' on, 'liga' off, 'clig' off",
      textDecorationThickness: '1px',
      textUnderlineOffset: '2px',
      '&:hover': {
        textDecorationThickness: '1px',
      },
    },
  },
  components: {
    // --- Portal design system: everything below reads the tokens in src/index.css ---
    MuiCssBaseline: {
      styleOverrides: (theme) => ({
        body: theme.palette.mode === 'light' ? { backgroundColor: 'var(--portal-canvas)' } : {},
      }),
    },
    MuiPaper: {
      styleOverrides: {
        rounded: { borderRadius: 'var(--portal-radius)' },
      },
    },
    MuiTableContainer: {
      styleOverrides: {
        // Internal scroll keeps the sticky header visible on long operational grids
        root: { maxHeight: 'var(--portal-table-max-height)' },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: ({ theme }) => ({
          height: 'var(--portal-row-height)',
          padding: 'var(--portal-cell-py) var(--portal-cell-px)',
          fontSize: 'var(--portal-cell-font-size)',
          lineHeight: 1.4,
          ...(theme.palette.mode === 'light' && { borderBottom: '1px solid var(--portal-border)' }),
        }),
        head: ({ theme }) => ({
          position: 'sticky',
          top: 0,
          zIndex: 2,
          fontSize: 'var(--portal-th-font-size)',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          whiteSpace: 'nowrap',
          backgroundColor: theme.palette.mode === 'light' ? 'var(--portal-th-bg)' : theme.palette.background.paper,
          color: theme.palette.mode === 'light' ? 'var(--portal-th-text)' : theme.palette.text.secondary,
        }),
      },
    },
    MuiTableRow: {
      styleOverrides: {
        hover: ({ theme }) => ({
          '&:hover': {
            backgroundColor: theme.palette.mode === 'light' ? 'var(--portal-row-hover) !important' : undefined,
          },
        }),
      },
    },
    MuiCardHeader: {
      styleOverrides: {
        root: ({ theme }) => ({
          '&[variant="heading"]': {
            backgroundColor: alpha(theme.palette.action.hover, 0.04),
            padding: '0.625rem 1rem',
            alignItems: 'center',
            '& .MuiCardHeader-title': {
              fontSize: '1.25rem',
              fontStyle: 'normal',
              fontWeight: 500,
              lineHeight: 1,
              letterSpacing: '-0.03125rem',
              marginBottom: '-0.25rem',
            },
          },
        }),
      },
    },
    MuiCard: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: 'var(--portal-radius)',
          border:
            theme.palette.mode === 'light' ? '1px solid var(--portal-border)' : `1px solid ${theme.palette.divider}`,
          background:
            theme.palette.mode === 'light' ? 'var(--portal-surface)' : theme.palette.background.paperElevation1,
          boxShadow: theme.palette.mode === 'light' ? 'var(--portal-shadow)' : 'none',
        }),
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: ({ theme }) => ({
          '&.MuiCardContent-variantHeading': {
            padding: '0.5rem 1rem',
            // child typography with variant="body2"
            '& .MuiTypography-body2': {
              color: theme.palette.text.secondary,
              fontSize: '0.875rem',
              fontStyle: 'normal',
              fontWeight: 400,
              lineHeight: '150%',
            },
          },
        }),
      },
    },
    // tabs overwrite
    MuiTab: {
      styleOverrides: {
        root: ({ theme }) => ({
          color: theme.palette.text.secondary,
          fontSize: '0.875rem',
          fontStyle: 'normal',
          fontWeight: 500,
          lineHeight: 1.4,
          textTransform: 'capitalize',
        }),
      },
    },
    MuiTypography: {
      styleOverrides: {
        subtitle2: {
          fontSize: '0.875rem',
          fontStyle: 'normal',
          fontWeight: 500,
          lineHeight: '150%',
        },
        h4: {
          fontWeight: 600,
          fontSize: '32px',
          lineHeight: 1.25,
          letterSpacing: '-0.5px',
        },
        h5: {
          fontWeight: 600,
          fontSize: '24px',
          lineHeight: 1.3,
        },
        body1: {
          fontSize: '16px',
          fontWeight: 400,
          lineHeight: 1.5,
        },
        body2: {
          fontSize: '14px',
          fontWeight: 400,
          lineHeight: 1.5,
        },
        subtitle1: {
          fontSize: '16px',
          fontWeight: 500,
          lineHeight: 1.5,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 'var(--portal-radius)',
          padding: '0.375rem 0.875rem',
          fontSize: '0.8125rem',
          fontStyle: 'normal',
          fontWeight: 500,
          lineHeight: '1.4',
          textTransform: 'capitalize',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
        outlinedInherit: ({ theme }) => ({
          border: `1px solid ${theme.palette.divider}`,
          color: theme.palette.text.primary,
          '&:hover': {
            backgroundColor: theme.palette.action.hover,
            border: `1px solid ${theme.palette.divider}`,
          },
        }),
        containedPrimary: ({ theme }) => ({
          background: theme.palette.primary.main,
          color: theme.palette.primary.contrastText,
          '&:hover': {
            background: theme.palette.primary.dark,
          },
        }),
      },
    },
    MuiLink: {
      styleOverrides: {
        root: {
          textDecorationThickness: '1px',
          textUnderlineOffset: '2px',
          '&:hover': {
            textDecorationThickness: '1px',
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: '0.5rem',
          padding: '0.375rem 1rem',
          fontSize: '0.875rem',
          fontWeight: 400,
          lineHeight: 1.5,
          '& .MuiAlert-icon': {
            fontSize: '1.375rem',
            marginRight: '0.75rem',
            padding: '0.4375rem 0',
          },
          '& .MuiAlert-message': {
            padding: '0.5rem 0',
            '& .MuiAlertTitle-root': {
              fontSize: '1rem',
              fontWeight: 500,
              lineHeight: 1.5,
              marginBottom: '0.25rem',
            },
          },
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: ({ theme, ownerState }) => ({
          borderRadius: ownerState.fullScreen ? 0 : '0.5rem',
          border: ownerState.fullScreen ? 'none' : `1px solid ${theme.palette.divider}`,
        }),
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '0.5rem',
          },
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: '0.5rem',
        },
      },
    },
  },
};

const baseTheme = createMuiTheme(themeOptions);

const modeBranchOptions = {
  dark: darkThemeOptions,
  light: lightThemeOptions,
  'high-contrast': highContrastThemeOptions,
};

export const createTheme = (config) => {
  const userMode = config?.palette?.mode ?? 'dark';
  const branch = modeBranchOptions[userMode] ?? darkThemeOptions;
  const muiPaletteMode = userMode === 'high-contrast' ? 'dark' : userMode === 'light' ? 'light' : 'dark';

  const resolvedPalette = {
    ...config.palette,
    mode: muiPaletteMode,
    ...(userMode === 'high-contrast' ? { highContrast: true } : { highContrast: false }),
  };

  const theme = deepmerge(deepmerge(baseTheme, branch), {
    ...config,
    palette: resolvedPalette,
  });
  return createMuiTheme(theme);
};
