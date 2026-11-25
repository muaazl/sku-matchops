import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { createBatch, fetchMerchantSkus } from '../services/batchApiService';
import { DOMAINS } from '../utils/constants';
import { buildBatchFormData, mapRows, parseCSV, parseExcelWorkbook } from '../utils/csvUtils';
import {
  isSupportedBulkFile,
  isSupportedCsvFile,
  readFileAsArrayBuffer,
  readFileAsText,
  stripFileExtension,
} from '../utils/fileUtils';
import { useAppStore } from './useAppStore';

export function useBatchProcess() {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const { merchantId, portalUrl, bearerToken, setMerchantCredentials, clearToken } = useAppStore();

  const [csvModalOpen, setCsvModalOpen] = useState(false);
  const [uploadMode, setUploadMode] = useState('file');
  const [domain, setDomain] = useState(DOMAINS[0]);
  const [task, setTask] = useState('pipeline');

  const [file, setFile] = useState(null);
  const [pastedText, setPastedText] = useState('');
  const [bulkFiles, setBulkFiles] = useState([]);
  const [pendingJobs, setPendingJobs] = useState(null);

  const [localMerchantId, setLocalMerchantId] = useState(merchantId);
  const [localPortalUrl] = useState(import.meta.env.VITE_PORTAL_URL || portalUrl || '');
  const [tokenModalOpen, setTokenModalOpen] = useState(false);
  const [localToken, setLocalToken] = useState(bearerToken || '');
  const [fetchingPortal, setFetchingPortal] = useState(false);

  const [previewData, setPreviewData] = useState(null);
  const [jobName, setJobName] = useState('');

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

  const fetchCsvFromPortal = async (tokenToUse) => {
    if (!localMerchantId.trim()) {
      enqueueSnackbar('Please enter a Merchant ID', { variant: 'warning' });
      return;
    }

    setFetchingPortal(true);
    try {
      const data = await fetchMerchantSkus({
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
        clearToken();
        setLocalToken('');
        enqueueSnackbar(
          err.response?.data?.detail ||
            'Your API token is invalid, blacklisted, or expired. The bad token has been cleared. ' +
              'Please enter a fresh token.',
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

  const triggerPortalFetch = () => {
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

  const submitTokenAndFetch = () => {
    setMerchantCredentials(localMerchantId.trim(), localPortalUrl.trim(), localToken.trim());
    setTokenModalOpen(false);
    fetchCsvFromPortal(localToken.trim());
  };

  const pickSingleFile = (selectedFile) => {
    if (!selectedFile) return;
    if (!isSupportedCsvFile(selectedFile.name)) {
      enqueueSnackbar('Please choose a .csv or .tsv file', { variant: 'warning' });
      return;
    }
    setFile(selectedFile);
  };

  const confirmCsvImport = async () => {
    if (uploadMode === 'file') {
      if (!file) {
        enqueueSnackbar('Please select a file to upload', { variant: 'warning' });
        return;
      }
      try {
        const text = await readFileAsText(file);
        const parsed = parseCSV(text);
        if (!parsed.rows || parsed.rows.length === 0) {
          enqueueSnackbar('Uploaded CSV has no data', { variant: 'error' });
          return;
        }
        const mapped = mapRows(parsed.headers, parsed.rows);
        const defaultName = stripFileExtension(file.name);
        setPreviewData({ rows: mapped, source: 'upload', name: defaultName, domain, task });
        setJobName(defaultName);
        setCsvModalOpen(false);
      } catch (err) {
        enqueueSnackbar(`Error parsing CSV: ${err.message}`, { variant: 'error' });
      }
      return;
    }

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
      setPreviewData({ rows: mapped, source: 'paste', name: 'Pasted SKU Job', domain, task });
      setJobName('Pasted SKU Job');
      setCsvModalOpen(false);
    } catch (err) {
      enqueueSnackbar(`Error parsing CSV: ${err.message}`, { variant: 'error' });
    }
  };

  const startSingleJob = () => {
    if (!jobName.trim()) {
      enqueueSnackbar('Please enter a job name', { variant: 'warning' });
      return;
    }
    createMutation.mutate(buildBatchFormData(previewData.rows, jobName.trim(), previewData.domain, previewData.task));
  };

  const cancelSinglePreview = () => {
    setPreviewData(null);
    setFile(null);
    setPastedText('');
  };

  const pickBulkFiles = (fileList) => {
    const incoming = Array.from(fileList || []);
    if (incoming.length === 0) return;
    const valid = incoming.filter((f) => isSupportedBulkFile(f.name));
    if (valid.length === 0) {
      enqueueSnackbar('Please choose CSV, TSV, or Excel (.xlsx/.xls) files', { variant: 'warning' });
      return;
    }
    if (valid.length < incoming.length) {
      enqueueSnackbar(`${incoming.length - valid.length} file(s) skipped (unsupported type)`, { variant: 'warning' });
    }
    setBulkFiles((prev) => [...prev, ...valid]);
  };

  const confirmBulkImport = async () => {
    if (bulkFiles.length === 0) {
      enqueueSnackbar('Please add at least one file', { variant: 'warning' });
      return;
    }

    const jobs = [];
    for (const f of bulkFiles) {
      const lower = f.name.toLowerCase();
      const baseName = stripFileExtension(f.name);
      try {
        if (lower.endsWith('.xlsx') || lower.endsWith('.xls')) {
          const buf = await readFileAsArrayBuffer(f);
          const sheets = await parseExcelWorkbook(buf);
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
      const data = await createBatch(buildBatchFormData(job.rows, job.name.trim() || job.id, domain, task));
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

  return {
    csvModalOpen,
    setCsvModalOpen,
    uploadMode,
    setUploadMode,
    domain,
    setDomain,
    task,
    setTask,
    file,
    setFile,
    pastedText,
    setPastedText,
    bulkFiles,
    setBulkFiles,
    pendingJobs,
    localMerchantId,
    setLocalMerchantId,
    tokenModalOpen,
    setTokenModalOpen,
    localToken,
    setLocalToken,
    fetchingPortal,
    previewData,
    jobName,
    setJobName,
    isCreatingBatch: createMutation.isPending,
    triggerPortalFetch,
    submitTokenAndFetch,
    pickSingleFile,
    confirmCsvImport,
    startSingleJob,
    cancelSinglePreview,
    pickBulkFiles,
    confirmBulkImport,
    updatePendingJob,
    startPendingJob,
    startAllPendingJobs,
    removePendingJob,
    navigateToJobs: () => navigate('/jobs'),
  };
}
