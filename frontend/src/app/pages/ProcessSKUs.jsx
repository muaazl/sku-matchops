import React, { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  Button,
  Stack,
  Chip,
  Tabs,
  Tab,
  Alert,
  InputAdornment,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  ToggleButtonGroup,
  ToggleButton,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
} from '@mui/material';
import { UploadCloud, FileText, X, Eye, EyeOff, ShieldCheck, Play } from 'lucide-react';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader } from '../components/ui';
import { DOMAINS } from '../constants';
import { useMutation } from '@tanstack/react-query';
import { createBatch } from '../api';
import { useStore } from '../store';
import { parseCSV, mapRows } from '../utils/csv';

export default function ProcessSKUs() {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const inputRef = useRef(null);

  // Zustand credentials
  const { merchantId, portalUrl, bearerToken, setMerchantCredentials, clearToken } = useStore();

  // Dialog State
  const [csvModalOpen, setCsvModalOpen] = useState(false);

  // Unified States
  const [uploadMode, setUploadMode] = useState('file'); // 'file' or 'paste' inside CSV modal
  const [domain, setDomain] = useState(DOMAINS[0]);
  const [task, setTask] = useState('pipeline');

  // File Upload State
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  // Paste CSV State
  const [pastedText, setPastedText] = useState('');

  // Merchant Fetch State
  const [localMerchantId, setLocalMerchantId] = useState(merchantId);
  const [localPortalUrl] = useState(import.meta.env.VITE_PORTAL_URL || portalUrl || '');
  const [tokenModalOpen, setTokenModalOpen] = useState(false);
  const [localToken, setLocalToken] = useState(bearerToken || '');
  const [showToken, setShowToken] = useState(false);
  const [fetchingPortal, setFetchingPortal] = useState(false);

  // Preview States
  const [previewData, setPreviewData] = useState(null);
  const [jobName, setJobName] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Reset pagination on data change
  useEffect(() => {
    setPage(0);
  }, [previewData]);

  // Mutations

  const createMutation = useMutation({
    mutationFn: createBatch,
    onSuccess: (data) => {
      enqueueSnackbar('CSV uploaded — job queued', { variant: 'success' });
      setCsvModalOpen(false);
      setPreviewData(null);
      setFile(null);
      setPastedText('');
      navigate(`/jobs/${data.job_id}`);
    },
    onError: (error) => {
      enqueueSnackbar(`Upload failed: ${error.message}`, { variant: 'error' });
    },
  });

  // Client-side fetch from Food Portal
  const fetchCsvFromPortal = async (tokenToUse) => {
    if (!localMerchantId.trim()) {
      enqueueSnackbar('Please enter a Merchant ID', { variant: 'warning' });
      return;
    }

    setFetchingPortal(true);
    const baseUrl =
      import.meta.env.VITE_PORTAL_URL || 'https://food-portal-api-go.pickme.lk/v1/food/place/skus/csv/{merchantid}';
    const url = baseUrl.replace('{merchantid}', localMerchantId.trim());

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          Authorization: 'Bearer ' + tokenToUse,
          'user-action': 'view_restaurant/view',
        },
      });

      const text = await response.text();

      // Try to parse error if it is JSON
      try {
        const json = JSON.parse(text);
        if (json && json.errors) {
          const isBlacklisted = json.errors.some(
            (err) =>
              err.code === 'MER-4007' ||
              err.code === 'MER-4006' ||
              (err.message &&
                (err.message.toLowerCase().includes('token blacklisted') ||
                  err.message.toLowerCase().includes('token expired')))
          );
          if (isBlacklisted) {
            clearToken();
            setLocalToken('');
            enqueueSnackbar(
              'Your API token has been blacklisted or expired. The bad token has been cleared. Please enter a fresh token.',
              { variant: 'error' }
            );
            setTokenModalOpen(true);
            setFetchingPortal(false);
            return;
          }

          const errorMsg = json.errors.map((e) => e.message || e.code).join(', ');
          throw new Error(errorMsg || 'Failed to fetch SKUs from portal');
        }
      } catch (e) {
        // Text is not JSON, proceed as CSV
      }

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}: ${response.statusText}`);
      }

      const parsed = parseCSV(text);
      if (!parsed.rows || parsed.rows.length === 0) {
        throw new Error('Fetched CSV has no data');
      }

      const mapped = mapRows(parsed.headers, parsed.rows);
      setPreviewData({
        rows: mapped,
        source: 'merchant',
        name: localMerchantId.trim(),
        domain,
        task,
      });
      setJobName(localMerchantId.trim());
    } catch (err) {
      enqueueSnackbar(`Portal Fetch Error: ${err.message}`, { variant: 'error' });
    } finally {
      setFetchingPortal(false);
    }
  };

  const handlePortalFetchTrigger = () => {
    if (!localMerchantId.trim()) {
      enqueueSnackbar('Please enter a Merchant ID', { variant: 'warning' });
      return;
    }
    if (bearerToken) {
      fetchCsvFromPortal(bearerToken);
    } else {
      setTokenModalOpen(true);
    }
  };

  const pick = (f) => {
    if (!f) return;
    const lower = f.name.toLowerCase();
    if (!lower.endsWith('.csv') && !lower.endsWith('.tsv') && !lower.endsWith('.txt')) {
      enqueueSnackbar('Please choose a .csv or .tsv file', { variant: 'warning' });
      return;
    }
    setFile(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    pick(e.dataTransfer.files?.[0]);
  };

  const handleCsvModalConfirm = () => {
    if (uploadMode === 'file') {
      if (!file) {
        enqueueSnackbar('Please select a file to upload', { variant: 'warning' });
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        try {
          const parsed = parseCSV(text);
          if (!parsed.rows || parsed.rows.length === 0) {
            enqueueSnackbar('Uploaded CSV has no data', { variant: 'error' });
            return;
          }
          const mapped = mapRows(parsed.headers, parsed.rows);
          const defaultName = file.name.replace(/\.[^/.]+$/, '');
          setPreviewData({
            rows: mapped,
            source: 'upload',
            name: defaultName,
            domain,
            task,
          });
          setJobName(defaultName);
          setCsvModalOpen(false);
        } catch (err) {
          enqueueSnackbar(`Error parsing CSV: ${err.message}`, { variant: 'error' });
        }
      };
      reader.readAsText(file);
    } else {
      if (!pastedText.trim()) {
        enqueueSnackbar('Please paste CSV content', { variant: 'warning' });
        return;
      }
      try {
        const parsed = parseCSV(pastedText);
        if (!parsed.rows || parsed.rows.length === 0) {
          enqueueSnackbar('Pasted CSV has no data', { variant: 'error' });
          return;
        }
        const mapped = mapRows(parsed.headers, parsed.rows);
        setPreviewData({
          rows: mapped,
          source: 'paste',
          name: 'Pasted SKU Job',
          domain,
          task,
        });
        setJobName('Pasted SKU Job');
        setCsvModalOpen(false);
      } catch (err) {
        enqueueSnackbar(`Error parsing CSV: ${err.message}`, { variant: 'error' });
      }
    }
  };

  const handleStartJob = () => {
    if (!jobName.trim()) {
      enqueueSnackbar('Please enter a job name', { variant: 'warning' });
      return;
    }

    const csvHeaders = ['name', 'price', 'description', 'category'];
    const csvLines = [csvHeaders.join(',')];

    previewData.rows.forEach((row) => {
      const line = csvHeaders.map((header) => {
        let val = String(row[header] || '');
        if (val.includes(',') || val.includes('"') || val.includes('\n') || val.includes('\r')) {
          val = `"${val.replace(/"/g, '""')}"`;
        }
        return val;
      });
      csvLines.push(line.join(','));
    });

    const csvContent = csvLines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const fileToUpload = new File([blob], jobName.trim(), { type: 'text/csv' });

    const formData = new FormData();
    formData.append('file', fileToUpload);
    formData.append('domain', previewData.domain);
    formData.append('task', previewData.task);
    formData.append('created_by', 'admin');

    createMutation.mutate(formData);
  };

  const handleTokenSubmit = () => {
    setMerchantCredentials(localMerchantId.trim(), localPortalUrl.trim(), localToken.trim());
    setTokenModalOpen(false);
    fetchCsvFromPortal(localToken.trim());
  };

  const canSubmitMerchant = localMerchantId.trim() !== '';

  const canSubmitCsv = uploadMode === 'file' ? file !== null : pastedText.trim().length > 0;

  // Render Table Preview Confirmation UI if previewData is set
  if (previewData) {
    const paginatedRows = previewData.rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

    return (
      <PageContainer>
        <PageHeader
          title="Confirm SKU Job"
          subtitle="Review and customize parsed SKU details before starting the batch processing job."
        />

        <Box sx={{ mt: 3 }}>
          <Card>
            <CardHeader
              title="Job Settings & Data Confirmation"
              titleTypographyProps={{ variant: 'h6', sx: { fontWeight: 600 } }}
              subheader={`Total SKU Items: ${previewData.rows.length}`}
            />
            <CardContent>
              <Stack spacing={3}>
                <Grid container spacing={2} alignItems="center">
                  <Grid size={{ xs: 12, md: 6 }}>
                    <TextField
                      label="Job / Sheet Name"
                      value={jobName}
                      onChange={(e) => setJobName(e.target.value)}
                      placeholder="e.g. mrc_8842"
                      fullWidth
                      required
                      error={!jobName.trim()}
                      helperText={!jobName.trim() ? 'Job name is required' : ''}
                    />
                  </Grid>
                  <Grid size={{ xs: 12, md: 6 }}>
                    <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                      <Chip
                        label={`Source: ${previewData.source}`}
                        color="info"
                        variant="outlined"
                        sx={{ textTransform: 'capitalize' }}
                      />
                      <Chip
                        label={`Domain: ${previewData.domain}`}
                        color="primary"
                        variant="outlined"
                        sx={{ textTransform: 'capitalize' }}
                      />
                      <Chip
                        label={`Task: ${previewData.task}`}
                        color="secondary"
                        variant="outlined"
                        sx={{ textTransform: 'capitalize' }}
                      />
                    </Stack>
                  </Grid>
                </Grid>

                <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                  Parsed SKUs Preview
                </Typography>

                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 440 }}>
                  <Table stickyHeader size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 600 }}>SKU Name</TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Category</TableCell>
                        <TableCell sx={{ fontWeight: 600 }} align="right">
                          Price
                        </TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Description</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {paginatedRows.map((row, idx) => (
                        <TableRow key={idx} hover>
                          <TableCell sx={{ fontWeight: 500 }}>{row.name || '—'}</TableCell>
                          <TableCell>{row.category || '—'}</TableCell>
                          <TableCell align="right">
                            {row.price !== undefined ? parseFloat(row.price).toFixed(2) : '0.00'}
                          </TableCell>
                          <TableCell
                            sx={{
                              maxWidth: 250,
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            }}
                          >
                            {row.description || '—'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                <TablePagination
                  rowsPerPageOptions={[5, 10, 25, 50]}
                  component="div"
                  count={previewData.rows.length}
                  rowsPerPage={rowsPerPage}
                  page={page}
                  onPageChange={(e, newPage) => setPage(newPage)}
                  onRowsPerPageChange={(e) => {
                    setRowsPerPage(parseInt(e.target.value, 10));
                    setPage(0);
                  }}
                />

                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                    gap: 2,
                    pt: 2,
                    borderTop: 1,
                    borderColor: 'divider',
                  }}
                >
                  <Button
                    variant="outlined"
                    color="inherit"
                    onClick={() => {
                      setPreviewData(null);
                      setFile(null);
                      setPastedText('');
                    }}
                    disabled={createMutation.isPending}
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={
                      createMutation.isPending ? <CircularProgress size={16} color="inherit" /> : <Play size={16} />
                    }
                    onClick={handleStartJob}
                    disabled={!jobName.trim() || createMutation.isPending}
                    sx={{ px: 4, fontWeight: 600 }}
                  >
                    {createMutation.isPending ? 'Starting...' : 'Start Processing'}
                  </Button>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Box>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Process SKUs"
        subtitle="Fetch product listings from a merchant portal or upload custom CSV data to kick off a SKU processing job."
      />

      <Grid container spacing={3} alignItems="flex-start" sx={{ mt: 1 }}>
        {/* Food Portal Integration Card */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Card>
            <CardHeader
              title="Portal Details"
              titleTypographyProps={{ variant: 'subtitle1', sx: { fontWeight: 600 } }}
            />
            <CardContent>
              <Stack spacing={2.5}>
                <TextField
                  label="Merchant ID"
                  value={localMerchantId}
                  onChange={(e) => setLocalMerchantId(e.target.value)}
                  placeholder="e.g. 8842"
                  fullWidth
                />

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                    Domain
                  </Typography>
                  <ToggleButtonGroup
                    size="small"
                    exclusive
                    value={domain}
                    onChange={(e, v) => v && setDomain(v)}
                    fullWidth
                  >
                    {DOMAINS.map((d) => (
                      <ToggleButton key={d} value={d} sx={{ textTransform: 'capitalize' }}>
                        {d}
                      </ToggleButton>
                    ))}
                  </ToggleButtonGroup>
                </Box>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                    Task
                  </Typography>
                  <ToggleButtonGroup size="small" exclusive value={task} onChange={(e, v) => v && setTask(v)} fullWidth>
                    <ToggleButton value="matcher" sx={{ textTransform: 'none' }}>
                      Matcher
                    </ToggleButton>
                    <ToggleButton value="classifier" sx={{ textTransform: 'none' }}>
                      Classifier
                    </ToggleButton>
                    <ToggleButton value="pipeline" sx={{ textTransform: 'none' }}>
                      Pipeline
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Box>

                <Button
                  variant="contained"
                  startIcon={fetchingPortal ? <CircularProgress size={16} color="inherit" /> : <Play size={16} />}
                  onClick={handlePortalFetchTrigger}
                  disabled={!canSubmitMerchant || fetchingPortal}
                  fullWidth
                  size="large"
                  sx={{ mt: 1, height: 42, fontWeight: 600 }}
                >
                  {fetchingPortal ? 'Fetching...' : 'Fetch & Process SKUs'}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* CSV Callout Column */}
        <Grid size={{ xs: 12, md: 7 }}>
          <Card
            sx={{
              p: 6,
              textAlign: 'center',
              border: '2px dashed',
              borderColor: 'divider',
              backgroundColor: 'action.hover',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: 400,
            }}
          >
            <Box sx={{ color: 'text.secondary', mb: 2 }}>
              <UploadCloud size={48} strokeWidth={1.5} />
            </Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
              Have a CSV to process?
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3, maxWidth: 360 }}>
              You can upload a CSV file or paste raw rows directly to match, classify, or execute pipelines.
            </Typography>

            {file ? (
              <Chip
                icon={<FileText size={15} />}
                label={`${file.name} · ${(file.size / 1024).toFixed(1)} KB`}
                onDelete={() => setFile(null)}
                deleteIcon={<X size={15} />}
                color="primary"
                variant="outlined"
                sx={{ mb: 3 }}
              />
            ) : pastedText.trim() ? (
              <Chip
                icon={<FileText size={15} />}
                label="Pasted SKU data ready"
                onDelete={() => setPastedText('')}
                deleteIcon={<X size={15} />}
                color="primary"
                variant="outlined"
                sx={{ mb: 3 }}
              />
            ) : null}

            <Button
              variant="contained"
              startIcon={<UploadCloud size={16} />}
              onClick={() => setCsvModalOpen(true)}
              size="large"
              sx={{ height: 42, px: 4, fontWeight: 600 }}
            >
              {file || pastedText.trim() ? 'Review & Submit CSV' : 'Upload or Paste CSV'}
            </Button>
          </Card>
        </Grid>
      </Grid>

      {/* CSV Import Modal (Dialog) */}
      <Dialog open={csvModalOpen} onClose={() => setCsvModalOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', pr: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Process CSV SKUs
          </Typography>
          <IconButton onClick={() => setCsvModalOpen(false)} size="small" aria-label="Close dialog">
            <X size={20} />
          </IconButton>
        </DialogTitle>

        <DialogContent dividers>
          <Tabs
            value={uploadMode}
            onChange={(e, val) => setUploadMode(val)}
            sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}
          >
            <Tab label="File Upload" value="file" />
            <Tab label="Paste CSV" value="paste" />
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
                onDrop={onDrop}
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
                <input ref={inputRef} type="file" accept=".csv,.tsv,.txt" hidden onChange={(e) => pick(e.target.files?.[0])} />
                <UploadCloud size={40} style={{ opacity: 0.6 }} />
                <Typography variant="subtitle1" sx={{ mt: 1 }}>
                  Drag & drop your CSV or TSV here
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  or click to browse
                </Typography>
              </Box>

              {file && (
                <Chip
                  icon={<FileText size={15} />}
                  label={`${file.name} · ${(file.size / 1024).toFixed(1)} KB`}
                  onDelete={() => setFile(null)}
                  deleteIcon={<X size={15} />}
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
                onChange={(e) => setPastedText(e.target.value)}
                sx={{
                  '& .MuiInputBase-input': {
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                  },
                }}
              />
            </Box>
          )}

          <Stack spacing={2.5} sx={{ mt: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                Domain
              </Typography>
              <ToggleButtonGroup size="small" exclusive value={domain} onChange={(e, v) => v && setDomain(v)} fullWidth>
                {DOMAINS.map((d) => (
                  <ToggleButton key={d} value={d} sx={{ textTransform: 'capitalize' }}>
                    {d}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
            </Box>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                Task
              </Typography>
              <ToggleButtonGroup size="small" exclusive value={task} onChange={(e, v) => v && setTask(v)} fullWidth>
                <ToggleButton value="matcher" sx={{ textTransform: 'none' }}>
                  Matcher
                </ToggleButton>
                <ToggleButton value="classifier" sx={{ textTransform: 'none' }}>
                  Classifier
                </ToggleButton>
                <ToggleButton value="pipeline" sx={{ textTransform: 'none' }}>
                  Pipeline
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
          </Stack>
        </DialogContent>

        <DialogActions sx={{ p: 2.5 }}>
          <Button onClick={() => setCsvModalOpen(false)} color="inherit">
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={!canSubmitCsv || createMutation.isPending}
            onClick={handleCsvModalConfirm}
            sx={{ px: 4 }}
          >
            {createMutation.isPending ? 'Processing…' : 'Process CSV'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Bearer Token Dialog */}
      <Dialog open={tokenModalOpen} onClose={() => setTokenModalOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Enter Bearer Token</DialogTitle>
        <DialogContent>
          <Alert severity="info" icon={<ShieldCheck size={20} />} sx={{ mb: 3, mt: 1 }}>
            The bearer token is kept in memory only — it is never stored or logged.
          </Alert>
          <TextField
            autoFocus
            label="Bearer token"
            type={showToken ? 'text' : 'password'}
            value={localToken}
            onChange={(e) => setLocalToken(e.target.value)}
            placeholder="••••••••••••"
            fullWidth
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowToken((s) => !s)}
                    edge="end"
                    size="small"
                    aria-label="Toggle token visibility"
                  >
                    {showToken ? <EyeOff size={17} /> : <Eye size={17} />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </DialogContent>
        <DialogActions sx={{ p: 2, pt: 0 }}>
          <Button onClick={() => setTokenModalOpen(false)} color="inherit">
            Cancel
          </Button>
          <Button variant="contained" onClick={handleTokenSubmit} disabled={!localToken.trim()}>
            Confirm & Fetch
          </Button>
        </DialogActions>
      </Dialog>
    </PageContainer>
  );
}
