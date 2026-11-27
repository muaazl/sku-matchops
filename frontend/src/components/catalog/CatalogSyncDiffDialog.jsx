import React from 'react';
import PropTypes from 'prop-types';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  List,
  ListItem,
  ListItemText,
  Stack,
  Typography,
} from '@mui/material';
import { Storage } from '@mui/icons-material';

const EMPTY_DOMAIN_DETAILS = {
  new_count: 0,
  changed_count: 0,
  new_rows: [],
  changed_rows: [],
};

export function CatalogSyncDiffDialog({ open, onClose, diffData, onConfirmSync }) {
  const market = diffData?.details?.market || EMPTY_DOMAIN_DETAILS;
  const food = diffData?.details?.food || EMPTY_DOMAIN_DETAILS;
  const totalNew = market.new_count + food.new_count;
  const totalChanged = market.changed_count + food.changed_count;
  const grandTotal = totalNew + totalChanged;
  const isCapped =
    market.is_capped ||
    food.is_capped ||
    market.new_count > market.new_rows?.length ||
    market.changed_count > market.changed_rows?.length ||
    food.new_count > food.new_rows?.length ||
    food.changed_count > food.changed_rows?.length;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
        },
      }}
    >
      <DialogTitle sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
        <Storage sx={{ fontSize: 20 }} /> Google Sheets Sync Status
      </DialogTitle>
      <DialogContent dividers>
        {diffData && (
          <Stack spacing={2}>
            {!diffData.has_changes ? (
              <Box sx={{ py: 3, textAlign: 'center' }}>
                <Typography variant="h6" color="success.main" sx={{ fontWeight: 600, mb: 1 }}>
                  ✓ Catalog is Up to Date
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  All local caches match the Google Sheets data perfectly. No synchronization is required.
                </Typography>
              </Box>
            ) : (
              <Box>
                <Alert severity="warning" sx={{ mb: 2.5 }}>
                  Found <strong>{grandTotal.toLocaleString()}</strong> total pending catalog updates (
                  <strong>{totalNew.toLocaleString()}</strong> new rows,{' '}
                  <strong>{totalChanged.toLocaleString()}</strong> modified rows).
                  {isCapped &&
                    ' Displaying top preview rows below. Synchronizing will apply all updates in the background.'}
                </Alert>

                {['market', 'food'].map((domainKey) => {
                  const domainDetails = diffData.details?.[domainKey] || EMPTY_DOMAIN_DETAILS;
                  if (domainDetails.new_count === 0 && domainDetails.changed_count === 0) return null;

                  const previewNew = (domainDetails.new_rows || []).slice(0, 20);
                  const previewChanged = (domainDetails.changed_rows || []).slice(0, 20);
                  const remainingNew = domainDetails.new_count - previewNew.length;
                  const remainingChanged = domainDetails.changed_count - previewChanged.length;

                  return (
                    <Card variant="outlined" key={domainKey} sx={{ mb: 2, borderRadius: '8px' }}>
                      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                        <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 1.5 }}>
                          <Typography
                            variant="subtitle2"
                            sx={{ fontWeight: 700, textTransform: 'uppercase', color: 'primary.main' }}
                          >
                            {domainKey === 'market' ? 'Market Domain' : 'Food Domain'}
                          </Typography>
                          <Stack direction="row" spacing={1}>
                            {domainDetails.new_count > 0 && (
                              <Chip
                                size="small"
                                label={`${domainDetails.new_count.toLocaleString()} New`}
                                color="success"
                                variant="outlined"
                                sx={{ fontWeight: 600 }}
                              />
                            )}
                            {domainDetails.changed_count > 0 && (
                              <Chip
                                size="small"
                                label={`${domainDetails.changed_count.toLocaleString()} Modified`}
                                color="warning"
                                variant="outlined"
                                sx={{ fontWeight: 600 }}
                              />
                            )}
                          </Stack>
                        </Stack>

                        {previewNew.length > 0 && (
                          <Box sx={{ mt: 1.5 }}>
                            <Typography
                              variant="caption"
                              display="block"
                              sx={{ fontWeight: 700, color: 'success.main', mb: 0.5 }}
                            >
                              NEW SKUs (showing {previewNew.length} of {domainDetails.new_count.toLocaleString()}):
                            </Typography>
                            <List size="small" dense sx={{ bgcolor: 'action.hover', borderRadius: '6px', py: 0.5 }}>
                              {previewNew.map((row, i) => (
                                <ListItem key={i} sx={{ py: 0.2 }}>
                                  <ListItemText
                                    primary={row.name}
                                    primaryTypographyProps={{ variant: 'body2', sx: { fontWeight: 500 } }}
                                    secondary={row.details.basictype ? `Basic Type: ${row.details.basictype}` : null}
                                  />
                                </ListItem>
                              ))}
                            </List>
                            {remainingNew > 0 && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ display: 'block', mt: 0.5, fontStyle: 'italic', pl: 1 }}
                              >
                                + {remainingNew.toLocaleString()} more new rows will be added during sync.
                              </Typography>
                            )}
                          </Box>
                        )}

                        {previewChanged.length > 0 && (
                          <Box sx={{ mt: 1.5 }}>
                            <Typography
                              variant="caption"
                              display="block"
                              sx={{ fontWeight: 700, color: 'warning.main', mb: 0.5 }}
                            >
                              MODIFIED SKUs (showing {previewChanged.length} of{' '}
                              {domainDetails.changed_count.toLocaleString()}):
                            </Typography>
                            <List size="small" dense sx={{ bgcolor: 'action.hover', borderRadius: '6px', py: 0.5 }}>
                              {previewChanged.map((row, i) => (
                                <ListItem key={i} sx={{ py: 0.5, alignItems: 'flex-start' }}>
                                  <Box sx={{ width: '100%' }}>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                      {row.name}
                                    </Typography>
                                    {Object.entries(row.diffs).map(([field, delta], j) => (
                                      <Typography
                                        key={j}
                                        variant="caption"
                                        display="block"
                                        color="text.secondary"
                                        sx={{ pl: 1 }}
                                      >
                                        • {field}:{' '}
                                        <span
                                          style={{
                                            textDecoration: 'line-through',
                                            color: '#ff8a8a',
                                            paddingRight: '4px',
                                          }}
                                        >
                                          &apos;{delta.old}&apos;
                                        </span>{' '}
                                        →{' '}
                                        <span style={{ color: '#8aff8a', fontWeight: 600 }}>
                                          &apos;{delta.new}&apos;
                                        </span>
                                      </Typography>
                                    ))}
                                  </Box>
                                </ListItem>
                              ))}
                            </List>
                            {remainingChanged > 0 && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ display: 'block', mt: 0.5, fontStyle: 'italic', pl: 1 }}
                              >
                                + {remainingChanged.toLocaleString()} more modified rows will be updated during sync.
                              </Typography>
                            )}
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </Box>
            )}
          </Stack>
        )}
      </DialogContent>
      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose} color="inherit">
          Close
        </Button>
        {diffData?.has_changes && (
          <Button variant="contained" color="primary" onClick={onConfirmSync}>
            Sync from Sheets
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}

CatalogSyncDiffDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  diffData: PropTypes.object,
  onConfirmSync: PropTypes.func.isRequired,
};
