import React from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Button,
  Chip,
  Divider,
  IconButton,
  MenuItem,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import { Add, DeleteOutline } from '@mui/icons-material';
import { CONDITION_LABELS, CONDITION_TYPES } from '../../utils/constants';
import { getConditionPlaceholder } from '../../utils/ruleUtils';
import TagAutocomplete from './TagAutocomplete';

export function RuleConditionsEditor({ conditions, conditionGroups, ruleDomain, onUpdateConditions }) {
  return (
    <Stack spacing={1.5}>
      {conditionGroups.map((group, gIdx) => (
        <React.Fragment key={`cond-group-${gIdx}`}>
          {gIdx > 0 && (
            <Divider sx={{ my: 0.5 }}>
              <Chip
                label="OR — alternative match"
                size="small"
                color="warning"
                variant="outlined"
                sx={{ fontWeight: 700, fontSize: '0.7rem', height: 22 }}
              />
            </Divider>
          )}
          <Stack
            spacing={1}
            sx={{
              p: 1.25,
              borderRadius: 2,
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: (theme) => alpha(theme.palette.action.hover, 0.03),
            }}
          >
            {group.map(({ cond: c, index: i }) => {
              const isNegated = !!c.negate;
              return (
                <Box
                  key={i}
                  sx={{
                    p: 1,
                    borderRadius: 1.5,
                    border: '1px solid',
                    borderColor: (theme) => (isNegated ? alpha(theme.palette.error.main, 0.4) : theme.palette.divider),
                    bgcolor: (theme) =>
                      isNegated ? alpha(theme.palette.error.main, 0.06) : theme.palette.background.paper,
                    transition: 'background-color 150ms ease, border-color 150ms ease',
                  }}
                >
                  <Stack
                    direction={{ xs: 'column', sm: 'row' }}
                    spacing={1}
                    alignItems={{ xs: 'stretch', sm: 'center' }}
                  >
                    <Stack direction="row" spacing={0.75} alignItems="center" sx={{ flexShrink: 0 }}>
                      {i === 0 ? (
                        <Box
                          sx={{
                            width: 82,
                            height: 32,
                            borderRadius: 1,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.72rem',
                            fontWeight: 700,
                            letterSpacing: 0.5,
                            color: 'primary.main',
                            bgcolor: (theme) => alpha(theme.palette.primary.main, 0.12),
                            border: (theme) => `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                          }}
                        >
                          IF
                        </Box>
                      ) : (
                        <Tooltip title="AND: must also match • OR: start alternative match">
                          <ToggleButtonGroup
                            exclusive
                            size="small"
                            value={c.operator || 'AND'}
                            onChange={(_, nextOp) => {
                              if (!nextOp) return;
                              const next = [...conditions];
                              next[i] = { ...c, operator: nextOp };
                              onUpdateConditions(next);
                            }}
                            aria-label={`Condition ${i + 1} operator`}
                            sx={{ height: 32, width: 82 }}
                          >
                            <ToggleButton
                              value="AND"
                              sx={{
                                flex: 1,
                                px: 0.5,
                                fontSize: '0.68rem',
                                fontWeight: 700,
                                '&.Mui-selected': {
                                  bgcolor: (theme) => alpha(theme.palette.primary.main, 0.16),
                                  color: 'primary.main',
                                },
                              }}
                            >
                              AND
                            </ToggleButton>
                            <ToggleButton
                              value="OR"
                              sx={{
                                flex: 1,
                                px: 0.5,
                                fontSize: '0.68rem',
                                fontWeight: 700,
                                '&.Mui-selected': {
                                  bgcolor: (theme) => alpha(theme.palette.warning.main, 0.2),
                                  color: 'warning.main',
                                },
                              }}
                            >
                              OR
                            </ToggleButton>
                          </ToggleButtonGroup>
                        </Tooltip>
                      )}

                      <Tooltip
                        title={
                          isNegated ? 'Negated: must NOT match (click to turn off)' : 'Click to invert (must NOT match)'
                        }
                      >
                        <Button
                          variant={isNegated ? 'contained' : 'outlined'}
                          color={isNegated ? 'error' : 'inherit'}
                          size="small"
                          aria-pressed={isNegated}
                          onClick={() => {
                            const next = [...conditions];
                            next[i] = { ...c, negate: isNegated ? 0 : 1 };
                            onUpdateConditions(next);
                          }}
                          sx={{
                            minWidth: 50,
                            height: 32,
                            px: 1,
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            borderStyle: isNegated ? 'solid' : 'dashed',
                            color: isNegated ? undefined : 'text.secondary',
                          }}
                        >
                          NOT
                        </Button>
                      </Tooltip>
                    </Stack>

                    <TextField
                      select
                      size="small"
                      value={c.condition_type}
                      onChange={(e) => {
                        const next = [...conditions];
                        next[i] = { ...c, condition_type: e.target.value };
                        onUpdateConditions(next);
                      }}
                      sx={{ width: { xs: '100%', sm: 200 }, flexShrink: 0 }}
                    >
                      {CONDITION_TYPES.map((t) => (
                        <MenuItem key={t} value={t}>
                          {CONDITION_LABELS[t] || t}
                        </MenuItem>
                      ))}
                    </TextField>

                    <Box sx={{ flex: 1, minWidth: 140, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <TagAutocomplete
                        value={c.value}
                        onChange={(newValue) => {
                          const next = [...conditions];
                          next[i] = { ...c, value: newValue };
                          onUpdateConditions(next);
                        }}
                        type={c.condition_type}
                        ruleDomain={ruleDomain}
                        placeholder={getConditionPlaceholder(c.condition_type)}
                      />
                      <Tooltip title="Remove condition">
                        <IconButton
                          aria-label="Delete condition"
                          size="small"
                          onClick={() => onUpdateConditions(conditions.filter((_, j) => j !== i))}
                        >
                          <DeleteOutline sx={{ fontSize: 16 }} />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Stack>
                </Box>
              );
            })}
          </Stack>
        </React.Fragment>
      ))}

      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        <Button
          size="small"
          variant="outlined"
          startIcon={<Add sx={{ fontSize: 14 }} />}
          onClick={() =>
            onUpdateConditions([
              ...conditions,
              {
                condition_group: 1,
                operator: 'AND',
                condition_type: 'sku_contains',
                value: '',
                negate: 0,
              },
            ])
          }
        >
          {conditions.length === 0 ? 'Add condition' : 'Add condition (AND)'}
        </Button>
        {conditions.length > 0 && (
          <Button
            size="small"
            variant="outlined"
            color="warning"
            startIcon={<Add sx={{ fontSize: 14 }} />}
            onClick={() =>
              onUpdateConditions([
                ...conditions,
                {
                  condition_group: 1,
                  operator: 'OR',
                  condition_type: 'sku_contains',
                  value: '',
                  negate: 0,
                },
              ])
            }
          >
            Add alternative (OR)
          </Button>
        )}
      </Stack>
    </Stack>
  );
}

RuleConditionsEditor.propTypes = {
  conditions: PropTypes.array.isRequired,
  conditionGroups: PropTypes.array.isRequired,
  ruleDomain: PropTypes.string.isRequired,
  onUpdateConditions: PropTypes.func.isRequired,
};
