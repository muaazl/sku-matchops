import React, { useRef, useState } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import { Close, CloudUpload, Description } from '@mui/icons-material';
import { DomainSelector, TaskSelector } from '../common/DomainTaskSelectors';

export function CsvImportDialog({
  open,
  onClose,
  uploadMode,
  onUploadModeChange,
  file,
  onPickSingleFile,
  onClearFile,
  pastedText,
  onPastedTextChange,
  bulkFiles,
  onPickBulkFiles,
  onRemoveBulkFile,
  domain,
  onDomainChange,
  task,
  onTaskChange,
  isSubmitting,
  onConfirmSingle,
  onConfirmBulk,
}) {
  const inputRef = useRef(null);
  const bulkInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [bulkDragOver, setBulkDragOver] = useState(false);

  const canSubmit =
    uploadMode === 'file' ? file !== null : uploadMode === 'bulk' ? bulkFiles.length > 0 : pastedText.trim().length > 0;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', pr: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Process CSV SKUs
        </Typography>
        <IconButton onClick={onClose} size="small" aria-label="Close dialog">
          <Close sx={{ fontSize: 20 }} />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers>
        <Tabs
          value={uploadMode}
          onChange={(e, val) => onUploadModeChange(val)}
          sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}
        >
          <Tab label="File Upload" value="file" />
          <Tab label="Paste CSV" value="paste" />
          <Tab label="Bulk Upload" value="bulk" />
        </Tabs>

        {uploadMode === 'file' && (
          <Box sx={{ mb: 3 }}>
            <Box
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                onPickSingleFile(e.dataTransfer.files?.[0]);
              }}
              sx={{
                border: (t) => `2px dashed ${dragOver ? t.palette.primary.main : t.palette.divider}`,
                borderRadius: 2,
                p: 5,
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all .15s',
                bgcolor: (t) => (dragOver ? t.palette.action.hover : 'transparent'),
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".csv,.tsv,.txt"
                hidden
                onChange={(e) => onPickSingleFile(e.target.files?.[0])}
              />
              <CloudUpload sx={{ fontSize: 40, opacity: 0.6 }} />
              <Typography variant="subtitle1" sx={{ mt: 1 }}>
                Drag &amp; drop your CSV or TSV here
              </Typography>
              <Typography variant="body2" color="text.secondary">
                or click to browse
              </Typography>
            </Box>

            {file && (
              <Chip
                icon={<Description sx={{ fontSize: 15 }} />}
                label={`${file.name} · ${(file.size / 1024).toFixed(1)} KB`}
                onDelete={onClearFile}
                deleteIcon={<Close sx={{ fontSize: 15 }} />}
                sx={{ mt: 2 }}
              />
            )}
          </Box>
        )}

        {uploadMode === 'paste' && (
          <Box sx={{ mb: 3 }}>
            <TextField
              multiline
              rows={8}
              variant="outlined"
              fullWidth
              placeholder="Paste your CSV/TSV content or spreadsheet cells here... (e.g. id,title,description,price)"
              value={pastedText}
              onChange={(e) => onPastedTextChange(e.target.value)}
              sx={{
                '& .MuiInputBase-input': {
                  fontFamily: 'monospace',
                  fontSize: '0.875rem',
                },
              }}
            />
          </Box>
        )}

        {uploadMode === 'bulk' && (
          <Box sx={{ mb: 3 }}>
            <Box
              onClick={() => bulkInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setBulkDragOver(true);
              }}
              onDragLeave={() => setBulkDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setBulkDragOver(false);
                onPickBulkFiles(e.dataTransfer.files);
              }}
              sx={{
                border: (t) => `2px dashed ${bulkDragOver ? t.palette.primary.main : t.palette.divider}`,
                borderRadius: 2,
                p: 5,
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all .15s',
                bgcolor: (t) => (bulkDragOver ? t.palette.action.hover : 'transparent'),
              }}
            >
              <input
                ref={bulkInputRef}
                type="file"
                accept=".csv,.tsv,.txt,.xlsx,.xls"
                multiple
                hidden
                onChange={(e) => {
                  onPickBulkFiles(e.target.files);
                  e.target.value = '';
                }}
              />
              <CloudUpload sx={{ fontSize: 40, opacity: 0.6 }} />
              <Typography variant="subtitle1" sx={{ mt: 1 }}>
                Drag &amp; drop multiple CSVs, or one Excel workbook
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Each CSV becomes one job; each tab of an Excel workbook becomes its own job
              </Typography>
            </Box>

            {bulkFiles.length > 0 && (
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 2 }}>
                {bulkFiles.map((f, idx) => (
                  <Chip
                    key={`${f.name}-${idx}`}
                    icon={<Description sx={{ fontSize: 15 }} />}
                    label={`${f.name} · ${(f.size / 1024).toFixed(1)} KB`}
                    onDelete={() => onRemoveBulkFile(idx)}
                    deleteIcon={<Close sx={{ fontSize: 15 }} />}
                  />
                ))}
              </Stack>
            )}
          </Box>
        )}

        <Stack spacing={2.5} sx={{ mt: 2 }}>
          <DomainSelector value={domain} onChange={onDomainChange} />
          <TaskSelector value={task} onChange={onTaskChange} />
        </Stack>
      </DialogContent>

      <DialogActions sx={{ p: 2.5 }}>
        <Button onClick={onClose} color="inherit">
          Cancel
        </Button>
        <Button
          variant="contained"
          disabled={!canSubmit || isSubmitting}
          onClick={uploadMode === 'bulk' ? onConfirmBulk : onConfirmSingle}
          sx={{ px: 4 }}
        >
          {isSubmitting
            ? 'Processing…'
            : uploadMode === 'bulk'
            ? `Review ${bulkFiles.length || ''} File${bulkFiles.length === 1 ? '' : 's'}`
            : 'Process CSV'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

CsvImportDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  uploadMode: PropTypes.string.isRequired,
  onUploadModeChange: PropTypes.func.isRequired,
  file: PropTypes.object,
  onPickSingleFile: PropTypes.func.isRequired,
  onClearFile: PropTypes.func.isRequired,
  pastedText: PropTypes.string.isRequired,
  onPastedTextChange: PropTypes.func.isRequired,
  bulkFiles: PropTypes.array.isRequired,
  onPickBulkFiles: PropTypes.func.isRequired,
  onRemoveBulkFile: PropTypes.func.isRequired,
  domain: PropTypes.string.isRequired,
  onDomainChange: PropTypes.func.isRequired,
  task: PropTypes.string.isRequired,
  onTaskChange: PropTypes.func.isRequired,
  isSubmitting: PropTypes.bool.isRequired,
  onConfirmSingle: PropTypes.func.isRequired,
  onConfirmBulk: PropTypes.func.isRequired,
};
