import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, IconButton, MenuItem, Stack, TextField, Tooltip } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { Add, DeleteOutline } from '@mui/icons-material';
import { ACTION_LABELS, ACTION_TYPES } from '../../utils/constants';
import TagAutocomplete from './TagAutocomplete';

export function RuleActionsEditor({ actions, ruleDomain, onUpdateActions }) {
  return (
    <Stack spacing={1.25}>
      {actions.map((a, i) => (
        <Box
          key={i}
          sx={{
            p: 1,
            borderRadius: 1.5,
            border: '1px solid',
            borderColor: 'divider',
            bgcolor: (theme) => theme.palette.background.paper,
          }}
        >
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ xs: 'stretch', sm: 'center' }}>
            <Box
              sx={{
                width: 56,
                height: 32,
                borderRadius: 1,
                flexShrink: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.7rem',
                fontWeight: 700,
                letterSpacing: 0.5,
                color: 'success.main',
                bgcolor: (theme) => alpha(theme.palette.success.main, 0.12),
                border: (theme) => `1px solid ${alpha(theme.palette.success.main, 0.3)}`,
              }}
            >
              {i === 0 ? 'THEN' : 'AND'}
            </Box>
            <TextField
              select
              size="small"
              value={a.action_type}
              onChange={(e) => {
                const next = [...actions];
                next[i] = { ...a, action_type: e.target.value };
                onUpdateActions(next);
              }}
              sx={{ width: { xs: '100%', sm: 215 }, flexShrink: 0 }}
            >
              {ACTION_TYPES.map((t) => (
                <MenuItem key={t} value={t}>
                  {ACTION_LABELS[t] || t}
                </MenuItem>
              ))}
            </TextField>
            <Box sx={{ flex: 1, minWidth: 140, display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <TagAutocomplete
                value={a.value}
                onChange={(newValue) => {
                  const next = [...actions];
                  next[i] = { ...a, value: newValue };
                  onUpdateActions(next);
                }}
                type={a.action_type}
                ruleDomain={ruleDomain}
                placeholder="Select or enter value..."
              />
              <Tooltip title="Remove action">
                <IconButton
                  aria-label="Delete action"
                  size="small"
                  onClick={() => onUpdateActions(actions.filter((_, j) => j !== i))}
                >
                  <DeleteOutline sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
            </Box>
          </Stack>
        </Box>
      ))}
      <Button
        size="small"
        variant="outlined"
        startIcon={<Add sx={{ fontSize: 14 }} />}
        sx={{ alignSelf: 'flex-start' }}
        onClick={() => onUpdateActions([...actions, { action_type: 'set_bt', value: '' }])}
      >
        Add action
      </Button>
    </Stack>
  );
}

RuleActionsEditor.propTypes = {
  actions: PropTypes.array.isRequired,
  ruleDomain: PropTypes.string.isRequired,
  onUpdateActions: PropTypes.func.isRequired,
};
