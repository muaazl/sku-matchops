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
import {
  CloudUpload,
  Description,
  Close,
  Visibility,
  VisibilityOff,
  VerifiedUser,
  PlayArrow,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader, IconModal, SectionCard } from '../components/ui';
import { DOMAINS } from '../constants';
import { useMutation } from '@tanstack/react-query';
import { createBatch, merchantFetch } from '../api';
import { useStore } from '../store';
import { parseCSV, mapRows, parseExcelWorkbook } from '../utils/csv';

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

  // Bulk Upload State (multiple CSVs and/or a multi-tab Excel workbook, queued together)
  const [bulkFiles, setBulkFiles] = useState([]);
  const [bulkDragOver, setBulkDragOver] = useState(false);
  const bulkInputRef = useRef(null);
  // Each entry: { id, name, sourceLabel, rows, status: 'pending'|'starting'|'started'|'error', jobId, error }
  const [pendingJobs, setPendingJobs] = useState(null);

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

  // Fetch from the Food Portal — the actual portal call happens server-side (the backend
  // holds the bearer token only for the duration of this one request; the browser never
  // talks to the portal directly).
  const fetchCsvFromPortal = async (tokenToUse) => {
    if (!localMerchantId.trim()) {
      enqueueSnackbar('Please enter a Merchant ID', { variant: 'warning' });
      return;
    }

    setFetchingPortal(true);
    try {
      const data = await merchantFetch({
        merchant_id: localMerchantId.trim(),
        bearer_token: tokenToUse,
        portal_url: localPortalUrl,
        domain,
        task,
      });

      const rows = data.rows || [];
      if (rows.length === 0) {
        throw new Error('Fetched CSV has no data');
      }

      setPreviewData({
        rows,
        source: 'merchant',
        name: localMerchantId.trim(),
        domain,
        task,
      });
      setJobName(localMerchantId.trim());
    } catch (err) {
      if (err.response?.status === 401) {
        // The backend confirmed this token is invalid, blacklisted, or expired — clear it
        // and ask for a fresh one, rather than only reacting to two specific portal error codes.
        clearToken();
        setLocalToken('');
        enqueueSnackbar(
          err.response?.data?.detail ||
            'Your API token is invalid, blacklisted, or expired. The bad token has been cleared. Please enter a fresh token.',
          { variant: 'error' }
        );
        setTokenModalOpen(true);
      } else {
        enqueueSnackbar(`Portal Fetch Error: ${err.response?.data?.detail || err.message}`, { variant: 'error' });
      }
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

  // Re-serializes normalized rows back into a CSV file for the existing /batches upload
  // endpoint — shared by the single-job flow and each card in the bulk-upload flow.
  const rowsToFormData = (rows, name, jobDomain, jobTask) => {
    const csvHeaders = ['name', 'price', 'description', 'category'];
    const csvLines = [csvHeaders.join(',')];

    rows.forEach((row) => {
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
    const fileToUpload = new File([blob], name, { type: 'text/csv' });

    const formData = new FormData();
    formData.append('file', fileToUpload);
    formData.append('domain', jobDomain);
    formData.append('task', jobTask);
    formData.append('created_by', 'admin');
    return formData;
  };

  const handleStartJob = () => {
    if (!jobName.trim()) {
      enqueueSnackbar('Please enter a job name', { variant: 'warning' });
      return;
    }
    createMutation.mutate(rowsToFormData(previewData.rows, jobName.trim(), previewData.domain, previewData.task));
  };

  // --- Bulk Upload (multiple CSVs and/or a multi-tab Excel workbook) ---

  const SUPPORTED_BULK_EXTS = ['.csv', '.tsv', '.txt', '.xlsx', '.xls'];

  const pickBulkFiles = (fileList) => {
    const incoming = Array.from(fileList || []);
    if (incoming.length === 0) return;
    const valid = incoming.filter((f) => SUPPORTED_BULK_EXTS.some((ext) => f.name.toLowerCase().endsWith(ext)));
    if (valid.length === 0) {
      enqueueSnackbar('Please choose CSV, TSV, or Excel (.xlsx/.xls) files', { variant: 'warning' });
      return;
    }
    if (valid.length < incoming.length) {
      enqueueSnackbar(`${incoming.length - valid.length} file(s) skipped (unsupported type)`, { variant: 'warning' });
    }
    setBulkFiles((prev) => [...prev, ...valid]);
  };

  const onBulkDrop = (e) => {
    e.preventDefault();
    setBulkDragOver(false);
    pickBulkFiles(e.dataTransfer.files);
  };

  const readFileAsText = (f) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsText(f);
    });

  const readFileAsArrayBuffer = (f) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(f);
    });

  const handleBulkConfirm = async () => {
    if (bulkFiles.length === 0) {
      enqueueSnackbar('Please add at least one file', { variant: 'warning' });
      return;
    }

    const jobs = [];
    for (const f of bulkFiles) {
      const lower = f.name.toLowerCase();
      const baseName = f.name.replace(/\.[^/.]+$/, '');
      try {
        if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
          const buf = await readFileAsArrayBuffer(f);
          const sheets = parseExcelWorkbook(buf);
          const sheetNames = Object.keys(sheets);
          if (sheetNames.length === 0) {
            jobs.push({
              id: `${f.name}::empty`,
              name: baseName,
              sourceLabel: f.name,
              rows: [],
              status: 'error',
              jobId: null,
              error: 'No data found in any sheet.',
            });
            continue;
          }
          sheetNames.forEach((sheetName) => {
            const { headers, rows } = sheets[sheetName];
            const mapped = mapRows(headers, rows);
            const multi = sheetNames.length > 1;
            jobs.push({
              id: `${f.name}::${sheetName}`,
              name: multi ? `${baseName} - ${sheetName}` : baseName,
              sourceLabel: multi ? `${f.name} / ${sheetName}` : f.name,
              rows: mapped,
              status: 'pending',
              jobId: null,
              error: null,
            });
          });
        } else {
          const text = await readFileAsText(f);
          const parsed = parseCSV(text);
          const mapped = mapRows(parsed.headers, parsed.rows);
          jobs.push({
            id: f.name,
            name: baseName,
            sourceLabel: f.name,
            rows: mapped,
            status: mapped.length > 0 ? 'pending' : 'error',
            jobId: null,
            error: mapped.length > 0 ? null : 'No data rows found.',
          });
        }
      } catch (err) {
        jobs.push({
          id: f.name,
          name: baseName,
          sourceLabel: f.name,
          rows: [],
          status: 'error',
          jobId: null,
          error: err.message,
        });
      }
    }

    setPendingJobs(jobs);
    setCsvModalOpen(false);
    setBulkFiles([]);
  };

  const updatePendingJob = (id, patch) => {
    setPendingJobs((prev) => (prev ? prev.map((j) => (j.id === id ? { ...j, ...patch } : j)) : prev));
  };

  const startPendingJob = async (job) => {
    if (job.status === 'starting' || job.status === 'started') return;
    updatePendingJob(job.id, { status: 'starting', error: null });
    try {
      const data = await createBatch(rowsToFormData(job.rows, job.name.trim() || job.id, domain, task));
      updatePendingJob(job.id, { status: 'started', jobId: data.job_id });
    } catch (err) {
      updatePendingJob(job.id, { status: 'error', error: err.response?.data?.detail || err.message });
    }
  };

  const startAllPendingJobs = () => {
    (pendingJobs || []).forEach((job) => {
      if (job.status === 'pending' || job.status === 'error') startPendingJob(job);
    });
  };

  const removePendingJob = (id) => {
    setPendingJobs((prev) => {
      const next = (prev || []).filter((j) => j.id !== id);
      return next.length > 0 ? next : null;
    });
  };

  const handleTokenSubmit = () => {
    setMerchantCredentials(localMerchantId.trim(), localPortalUrl.trim(), localToken.trim());
    setTokenModalOpen(false);
    fetchCsvFromPortal(localToken.trim());
  };

  const canSubmitMerchant = localMerchantId.trim() !== '';

  const canSubmitCsv =
    uploadMode === 'file' ? file !== null : uploadMode === 'bulk' ? bulkFiles.length > 0 : pastedText.trim().length > 0;

  // Render bulk-upload queue (multiple CSVs / Excel tabs) if pendingJobs is set
  if (pendingJobs) {
    const anyPending = pendingJobs.some((j) => j.status === 'pending' || j.status === 'error');
    return (
      <PageContainer>
        <PageHeader
          title="Bulk SKU Jobs"
          subtitle="Review each file/sheet below, then start them individually or all at once."
        />

        <Stack spacing={2} sx={{ mt: 3 }}>
          {pendingJobs.map((job) => (
            <Card key={job.id} variant="outlined">
              <CardContent>
                <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
                  <Box sx={{ flexGrow: 1, minWidth: 200 }}>
                    <TextField
                      value={job.name}
                      onChange={(e) => updatePendingJob(job.id, { name: e.target.value })}
                      label="Job name"
                      size="small"
                      fullWidth
                      disabled={job.status === 'starting' || job.status === 'started'}
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                      {job.sourceLabel} · {job.rows.length} SKU{job.rows.length === 1 ? '' : 's'}
                    </Typography>
                  </Box>

                  {job.status === 'started' ? (
                    <Chip label={`Queued — Job ${job.jobId}`} color="success" size="small" />
                  ) : job.status === 'error' ? (
                    <Chip label={job.error || 'Failed'} color="error" size="small" sx={{ maxWidth: 320 }} />
                  ) : job.status === 'starting' ? (
                    <Chip icon={<CircularProgress size={12} color="inherit" />} label="Starting…" size="small" />
                  ) : job.rows.length === 0 ? (
                    <Chip label="No rows" color="warning" size="small" />
                  ) : null}

                  <Stack direction="row" spacing={1}>
                    <Button
                      size="small"
                      variant="contained"
                      disabled={job.status === 'starting' || job.status === 'started' || job.rows.length === 0}
                      onClick={() => startPendingJob(job)}
                    >
                      {job.status === 'error' ? 'Retry' : 'Start'}
                    </Button>
                    <Button
                      size="small"
                      color="inherit"
                      variant="outlined"
                      disabled={job.status === 'starting'}
                      onClick={() => removePendingJob(job.id)}
                    >
                      {job.status === 'started' ? 'Dismiss' : 'Remove'}
                    </Button>
                  </Stack>
                </Stack>
              </CardContent>
            </Card>
          ))}

          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1.5, pt: 1 }}>
            <Button color="inherit" onClick={() => navigate('/jobs')}>
              Done
            </Button>
            <Button variant="contained" disabled={!anyPending} onClick={startAllPendingJobs}>
              Start All
            </Button>
          </Box>
        </Stack>
      </PageContainer>
    );
  }

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
                      createMutation.isPending ? (
                        <CircularProgress size={16} color="inherit" />
                      ) : (
                        <PlayArrow sx={{ fontSize: 16 }} />
                      )
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
          <SectionCard title="Portal Details" subheader="Fetch live merchant catalog via portal integration">
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
                  aria-label="Select domain"
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
                <ToggleButtonGroup
                  size="small"
                  exclusive
                  value={task}
                  onChange={(e, v) => v && setTask(v)}
                  aria-label="Select matching task"
                  fullWidth
                >
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
                startIcon={
                  fetchingPortal ? <CircularProgress size={16} color="inherit" /> : <PlayArrow sx={{ fontSize: 16 }} />
                }
                onClick={handlePortalFetchTrigger}
                disabled={!canSubmitMerchant || fetchingPortal}
                fullWidth
                size="large"
                sx={{ mt: 1, height: 42, fontWeight: 600 }}
              >
                {fetchingPortal ? 'Fetching...' : 'Fetch & Process SKUs'}
              </Button>
            </Stack>
          </SectionCard>
        </Grid>

        {/* CSV Callout Column */}
        <Grid size={{ xs: 12, md: 7 }}>
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
                onDelete={() => setFile(null)}
                deleteIcon={<Close sx={{ fontSize: 15 }} />}
                color="primary"
                variant="outlined"
                sx={{ mb: 3, fontWeight: 500 }}
              />
            ) : pastedText.trim() ? (
              <Chip
                icon={<Description sx={{ fontSize: 15 }} />}
                label="Pasted SKU data ready"
                onDelete={() => setPastedText('')}
                deleteIcon={<Close sx={{ fontSize: 15 }} />}
                color="primary"
                variant="outlined"
                sx={{ mb: 3, fontWeight: 500 }}
              />
            ) : null}

            <Button
              variant="contained"
              startIcon={<CloudUpload sx={{ fontSize: 16 }} />}
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
            <Close sx={{ fontSize: 20 }} />
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
                <input
                  ref={inputRef}
                  type="file"
                  accept=".csv,.tsv,.txt"
                  hidden
                  onChange={(e) => pick(e.target.files?.[0])}
                />
                <CloudUpload sx={{ fontSize: 40, opacity: 0.6 }} />
                <Typography variant="subtitle1" sx={{ mt: 1 }}>
                  Drag & drop your CSV or TSV here
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  or click to browse
                </Typography>
              </Box>

              {file && (
                <Chip
                  icon={<Description sx={{ fontSize: 15 }} />}
                  label={`${file.name} · ${(file.size / 1024).toFixed(1)} KB`}
                  onDelete={() => setFile(null)}
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

          {uploadMode === 'bulk' && (
            <Box sx={{ mb: 3 }}>
              <Box
                onClick={() => bulkInputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setBulkDragOver(true);
                }}
                onDragLeave={() => setBulkDragOver(false)}
                onDrop={onBulkDrop}
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
                    pickBulkFiles(e.target.files);
                    e.target.value = '';
                  }}
                />
                <CloudUpload sx={{ fontSize: 40, opacity: 0.6 }} />
                <Typography variant="subtitle1" sx={{ mt: 1 }}>
                  Drag & drop multiple CSVs, or one Excel workbook
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
                      onDelete={() => setBulkFiles((prev) => prev.filter((_, i) => i !== idx))}
                      deleteIcon={<Close sx={{ fontSize: 15 }} />}
                    />
                  ))}
                </Stack>
              )}
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
            onClick={uploadMode === 'bulk' ? handleBulkConfirm : handleCsvModalConfirm}
            sx={{ px: 4 }}
          >
            {createMutation.isPending
              ? 'Processing…'
              : uploadMode === 'bulk'
              ? `Review ${bulkFiles.length || ''} File${bulkFiles.length === 1 ? '' : 's'}`
              : 'Process CSV'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Bearer Token Modal */}
      <IconModal
        open={tokenModalOpen}
        onClose={() => setTokenModalOpen(false)}
        icon={VerifiedUser}
        title="Enter Bearer Token"
        description="The bearer token is kept only for this browser tab's session (never written to disk) and is cleared automatically when you close the tab, or immediately if the portal reports it as invalid, blacklisted, or expired."
        actions={
          <>
            <Button onClick={() => setTokenModalOpen(false)} color="inherit" variant="outlined" fullWidth>
              Cancel
            </Button>
            <Button variant="contained" fullWidth onClick={handleTokenSubmit} disabled={!localToken.trim()}>
              Confirm & Fetch
            </Button>
          </>
        }
      >
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
                  {showToken ? <VisibilityOff sx={{ fontSize: 17 }} /> : <Visibility sx={{ fontSize: 17 }} />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
      </IconModal>
    </PageContainer>
  );
}
