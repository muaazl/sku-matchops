import React from 'react';
import PropTypes from 'prop-types';
import { Button, CircularProgress, FormControlLabel, Stack, Switch, TextField, Typography } from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import { DomainSelector, TaskSelector } from '../common/DomainTaskSelectors';
import { SectionCard } from '../common/SectionCard';

export function InteractiveInputForm({
  sku,
  onSkuChange,
  price,
  onPriceChange,
  category,
  onCategoryChange,
  description,
  onDescriptionChange,
  domain,
  onDomainChange,
  task,
  onTaskChange,
  enableSuggestions,
  onToggleSuggestions,
  loading,
  onRun,
}) {
  return (
    <SectionCard title="Input Details" subheader="Configure test SKU parameters">
      <Stack spacing={2.5}>
        <TextField
          label="SKU name"
          value={sku}
          onChange={(e) => onSkuChange(e.target.value)}
          fullWidth
          onKeyDown={(e) => e.key === 'Enter' && !loading && onRun()}
        />
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
          <TextField label="Price" value={price} onChange={(e) => onPriceChange(e.target.value)} fullWidth />
          <TextField
            label="Seller Category"
            value={category}
            onChange={(e) => onCategoryChange(e.target.value)}
            fullWidth
          />
        </Stack>
        <TextField
          label="Description"
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          fullWidth
          multiline
          rows={2}
        />

        <Stack spacing={2}>
          <DomainSelector value={domain} onChange={onDomainChange} />
          <TaskSelector value={task} onChange={onTaskChange} />

          <FormControlLabel
            control={
              <Switch
                checked={enableSuggestions}
                onChange={(e) => onToggleSuggestions(e.target.checked)}
                color="primary"
                size="small"
              />
            }
            label={
              <Typography variant="body2" color="text.secondary" fontWeight={500}>
                Enable Tag Suggestions (Experimental)
              </Typography>
            }
            sx={{ mt: 0.5 }}
          />
        </Stack>

        <Button
          variant="contained"
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <PlayArrow sx={{ fontSize: 16 }} />}
          onClick={onRun}
          disabled={loading}
          fullWidth
          size="large"
          sx={{ mt: 1, height: 42, fontWeight: 600 }}
        >
          {loading ? 'Running...' : 'Run'}
        </Button>
      </Stack>
    </SectionCard>
  );
}

InteractiveInputForm.propTypes = {
  sku: PropTypes.string.isRequired,
  onSkuChange: PropTypes.func.isRequired,
  price: PropTypes.string.isRequired,
  onPriceChange: PropTypes.func.isRequired,
  category: PropTypes.string.isRequired,
  onCategoryChange: PropTypes.func.isRequired,
  description: PropTypes.string.isRequired,
  onDescriptionChange: PropTypes.func.isRequired,
  domain: PropTypes.string.isRequired,
  onDomainChange: PropTypes.func.isRequired,
  task: PropTypes.string.isRequired,
  onTaskChange: PropTypes.func.isRequired,
  enableSuggestions: PropTypes.bool.isRequired,
  onToggleSuggestions: PropTypes.func.isRequired,
  loading: PropTypes.bool.isRequired,
  onRun: PropTypes.func.isRequired,
};
