import React from 'react';
import PropTypes from 'prop-types';
import { Box, Button, Card, Chip, Typography } from '@mui/material';
import { Close, CloudUpload, Description } from '@mui/icons-material';

export function CsvIntakeCalloutCard({ file, pastedText, onClearFile, onClearPastedText, onOpenModal }) {
  const hasContent = Boolean(file || pastedText.trim());

  return (
    <Card
      sx={{
        p: 6,
        textAlign: 'center',
        border: '2px dashed',
        borderColor: 'divider',
        borderRadius: 2,
        backgroundColor: 'action.hover',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 400,
        transition: 'border-color 150ms ease, box-shadow 150ms ease',
        '&:hover': {
          borderColor: 'primary.main',
        },
      }}
    >
      <Box
        sx={{
          width: 64,
          height: 64,
          borderRadius: '50%',
          bgcolor: (t) => (t.palette.mode === 'dark' ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)'),
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'primary.main',
          mb: 2,
        }}
      >
        <CloudUpload sx={{ fontSize: 36 }} />
      </Box>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
        Have a CSV to process?
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3, maxWidth: 380, lineHeight: 1.6 }}>
        Upload a CSV, paste raw rows, or queue multiple CSVs / an Excel workbook at once via Bulk Upload to match,
        classify, or execute pipelines.
      </Typography>

      {file ? (
        <Chip
          icon={<Description sx={{ fontSize: 15 }} />}
          label={`${file.name} · ${(file.size / 1024).toFixed(1)} KB`}
          onDelete={onClearFile}
          deleteIcon={<Close sx={{ fontSize: 15 }} />}
          color="primary"
          variant="outlined"
          sx={{ mb: 3, fontWeight: 500 }}
        />
      ) : pastedText.trim() ? (
        <Chip
          icon={<Description sx={{ fontSize: 15 }} />}
          label="Pasted SKU data ready"
          onDelete={onClearPastedText}
          deleteIcon={<Close sx={{ fontSize: 15 }} />}
          color="primary"
          variant="outlined"
          sx={{ mb: 3, fontWeight: 500 }}
        />
      ) : null}

      <Button
        variant="contained"
        startIcon={<CloudUpload sx={{ fontSize: 16 }} />}
        onClick={onOpenModal}
        size="large"
        sx={{ height: 42, px: 4, fontWeight: 600 }}
      >
        {hasContent ? 'Review & Submit CSV' : 'Upload or Paste CSV'}
      </Button>
    </Card>
  );
}

CsvIntakeCalloutCard.propTypes = {
  file: PropTypes.object,
  pastedText: PropTypes.string.isRequired,
  onClearFile: PropTypes.func.isRequired,
  onClearPastedText: PropTypes.func.isRequired,
  onOpenModal: PropTypes.func.isRequired,
};
