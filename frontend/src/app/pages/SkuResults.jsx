import React, { useMemo, useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  TextField,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Button,
  Stack,
  Table,
  TableCell,
  TableRow,
  TablePagination,
  IconButton,
} from '@mui/material';
import { Download, ArrowLeft, Info, ChevronDown, FileSpreadsheet, ShieldCheck } from 'lucide-react';
import { useSnackbar } from 'notistack';
import { PageContainer, PageHeader, SideDrawer, ConfirmDialog } from '../components/ui';
import RulesAppliedList from '../components/RulesAppliedList';
import { DOMAINS } from '../constants';
import { fmtTime, download, toCsv, formatGk, formatGkText } from '../utils';
import { useQuery } from '@tanstack/react-query';
import { getProcessedSkus, getJobs, retryJob } from '../api';
import { useDebounce } from '../hooks/useDebounce';
import {
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledHeaderCell,
  StyledTableRow,
  TableSkeleton,
} from '../../components/Common/StyledTable';

export default function SkuResults() {
  const { enqueueSnackbar } = useSnackbar();
  const location = useLocation();
  const navigate = useNavigate();
  const navigatedJobId = location.state?.jobId;

  // View state: null = Jobs view, string = SKUs view for a specific job/batch ID
  const [selectedJob, setSelectedJob] = useState(null);

  // Jobs view filters
  const [domainFilter, setDomainFilter] = useState('all');
  const [sheetFilter, setSheetFilter] = useState('');
  const [taskFilter, setTaskFilter] = useState('all');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // SKUs view state
  const [selectedSku, setSelectedSku] = useState(null);

  // State for job that needs rerun if not fully processed
  const [rerunJob, setRerunJob] = useState(null);

  // SKUs view pagination state
  const [skuPage, setSkuPage] = useState(0);
  const [skuRowsPerPage, setSkuRowsPerPage] = useState(100);

  // SKUs view search & filter states
  const [skuSearch, setSkuSearch] = useState('');
  const [skuBtFilter, setSkuBtFilter] = useState('');
  const [skuGkFilter, setSkuGkFilter] = useState('');
  const [skuSourceFilter, setSkuSourceFilter] = useState('all');

  // Debounce search/filter terms to prevent heavy array filtering on every keystroke
  const debouncedSkuSearch = useDebounce(skuSearch, 300);
  const debouncedSkuBtFilter = useDebounce(skuBtFilter, 300);
  const debouncedSkuGkFilter = useDebounce(skuGkFilter, 300);

  // Export menu anchor state
  const [exportAnchorEl, setExportAnchorEl] = useState(null);

  // Reset SKU page and filters when selected job changes
  useEffect(() => {
    setSkuPage(0);
    setSkuSearch('');
    setSkuBtFilter('');
    setSkuGkFilter('');
    setSkuSourceFilter('all');
    setExportAnchorEl(null);
  }, [selectedJob]);

  // Fetch Jobs
  const { data: serverJobs = [], isLoading: isLoadingJobs } = useQuery({
    queryKey: ['jobs_processed'],
    queryFn: () => getJobs({}),
  });

  useEffect(() => {
    if (navigatedJobId && serverJobs && serverJobs.length > 0) {
      const job = serverJobs.find((j) => String(j.id) === String(navigatedJobId));
      if (job) {
        if (job.status !== 'completed') {
          setRerunJob(job);
        } else {
          setSelectedJob(job);
        }
        // Clear the state so navigating back/away resets clean
        navigate(location.pathname, { replace: true, state: {} });
      }
    }
  }, [navigatedJobId, serverJobs, navigate, location.pathname]);

  const jobsList = useMemo(() => {
    let sorted = [...serverJobs].sort((a, b) => new Date(b.started_at || 0) - new Date(a.started_at || 0));
    // Only show completed jobs in the results section
    sorted = sorted.filter((j) => j.status === 'completed');
    if (domainFilter !== 'all') {
      sorted = sorted.filter((j) => j.domain === domainFilter);
    }
    if (sheetFilter.trim()) {
      sorted = sorted.filter((j) => {
        const name = j.target_sheet || j.sheet_name || '';
        return name.toLowerCase().includes(sheetFilter.toLowerCase());
      });
    }
    if (taskFilter !== 'all') {
      sorted = sorted.filter((j) => j.task?.toLowerCase() === taskFilter.toLowerCase());
    }
    return sorted;
  }, [serverJobs, domainFilter, sheetFilter, taskFilter]);

  const paginatedJobs = useMemo(() => {
    return jobsList.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [jobsList, page, rowsPerPage]);

  // Fetch SKUs for selected Job
  const { data: serverSkus = [], isLoading: isLoadingSkus } = useQuery({
    queryKey: ['processed_skus', selectedJob?.id],
    queryFn: () => getProcessedSkus({ batch_id: selectedJob?.batch_id || selectedJob?.id, limit: 0 }),
    enabled: !!selectedJob,
  });

  // Pre-process and normalize fields once per data fetch to avoid JSON.parse and string transforms inside filter/render loops
  const processedSkus = useMemo(() => {
    return serverSkus.map((row) => {
      const formattedGk = formatGk(row.generic_keywords || row.gk_json || row.gk);
      const cleanGk = formatGkText(row.generic_keywords || row.gk_json || row.gk);
      const btVal = row.basic_type || row.bt || '';
      const skuName = row.sku_name || '';
      return {
        ...row,
        _formattedGk: formattedGk,
        _cleanGk: cleanGk,
        _normName: skuName.toLowerCase(),
        _normBt: btVal.toLowerCase(),
        _normGk: formattedGk.toLowerCase(),
        _normSource: (row.match_source || '').toLowerCase(),
      };
    });
  }, [serverSkus]);

  const skusList = useMemo(() => {
    let filtered = processedSkus;

    // Filter by SKU Name (case-insensitive)
    if (debouncedSkuSearch.trim()) {
      const term = debouncedSkuSearch.toLowerCase().trim();
      filtered = filtered.filter((row) => row._normName.includes(term));
    }

    // Filter by Basic Type (BT, case-insensitive)
    if (debouncedSkuBtFilter.trim()) {
      const term = debouncedSkuBtFilter.toLowerCase().trim();
      filtered = filtered.filter((row) => row._normBt.includes(term));
    }

    // Filter by Generic Keywords (GK, case-insensitive)
    if (debouncedSkuGkFilter.trim()) {
      const term = debouncedSkuGkFilter.toLowerCase().trim();
      filtered = filtered.filter((row) => row._normGk.includes(term));
    }

    // Filter by Source (matcher / classifier)
    if (skuSourceFilter !== 'all') {
      filtered = filtered.filter((row) => {
        if (skuSourceFilter === 'classifier') {
          return row._normSource.includes('classifier');
        } else if (skuSourceFilter === 'matcher') {
          return row._normSource.includes('matcher') || row._normSource.includes('catalogue');
        }
        return true;
      });
    }

    return filtered;
  }, [processedSkus, debouncedSkuSearch, debouncedSkuBtFilter, debouncedSkuGkFilter, skuSourceFilter]);

  const paginatedSkus = useMemo(() => {
    return skusList.slice(skuPage * skuRowsPerPage, skuPage * skuRowsPerPage + skuRowsPerPage);
  }, [skusList, skuPage, skuRowsPerPage]);

  const exportRows = (type = 'clean') => {
    if (!selectedJob) return;
    if (!skusList.length) {
      enqueueSnackbar('No SKU data available to export', { variant: 'warning' });
      return;
    }

    const rawSheetName = selectedJob.target_sheet || selectedJob.sheet_name || 'Sheet';
    const baseSheetName = rawSheetName.replace(/\.[^/.]+$/, '');
    const safeSheetName = baseSheetName.replace(/[/\\?%*:|"<>]/g, '_').trim() || 'Sheet';

    if (type === 'clean') {
      const cleanRows = skusList.map((row) => {
        const cleanGk = row._cleanGk ?? formatGkText(row.generic_keywords || row.gk_json || row.gk);
        if (selectedJob.domain === 'market') {
          return {
            'SKU Name': row.sku_name || '',
            Categories: row.categories || row.region || '',
            'Generic Keywords': cleanGk,
            'Basic Type': row.basic_type || row.bt || '',
          };
        } else if (selectedJob.domain === 'food') {
          return {
            'SKU Name': row.sku_name || '',
            'Generic Keywords': cleanGk,
            'Basic Type': row.basic_type || row.bt || '',
            Region: row.region || '',
          };
        } else {
          return {
            'SKU Name': row.sku_name || '',
            Categories: row.categories || '',
            'Generic Keywords': cleanGk,
            'Basic Type': row.basic_type || row.bt || '',
            Region: row.region || '',
          };
        }
      });

      const filename = `Job_${selectedJob.id}_${safeSheetName}_clean.csv`;
      download(filename, toCsv(cleanRows));
      enqueueSnackbar(`Exported ${cleanRows.length} rows to ${filename} (Clean Version)`, { variant: 'success' });
    } else if (type === 'audit') {
      const auditRows = skusList.map((row) => ({
        ID: row.id,
        'Batch ID': row.batch_id,
        Domain: row.domain,
        'SKU Name': row.sku_name || '',
        'Generic Keywords': row._cleanGk ?? formatGkText(row.generic_keywords || row.gk_json || row.gk),
        'Basic Type': row.basic_type || row.bt || '',
        Region: row.region || '',
        Categories: row.categories || '',
        'Match Source': row.match_source || '',
        'Matched Catalog Name': row.matched_catalog_name || '',
        'Match Score': row.match_score != null ? row.match_score : '',
        'BT Confidence': row.bt_confidence != null ? row.bt_confidence : '',
        'GK Confidence': row.gk_confidence != null ? row.gk_confidence : '',
        'Region Confidence': row.region_confidence != null ? row.region_confidence : '',
        'Final Confidence': row.confidence != null ? row.confidence : '',
        'Logic Notes': row.logic_notes || '',
        'Rules Applied':
          typeof row.rules_applied_json === 'object'
            ? JSON.stringify(row.rules_applied_json)
            : row.rules_applied_json || '',
        'Created At': row.created_at || '',
      }));

      const filename = `Job_${selectedJob.id}_${safeSheetName}_audit.csv`;
      download(filename, toCsv(auditRows));
      enqueueSnackbar(`Exported ${auditRows.length} rows to ${filename} (Audit Version)`, { variant: 'success' });
    }
  };

  // -------------------------
  // JOBS VIEW
  // -------------------------
  if (!selectedJob) {
    return (
      <PageContainer>
        <PageHeader title="SKU Results" subtitle="Select a job to view its SKU results." />

        <Card sx={{ p: 2, mb: 3 }}>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems={{ md: 'center' }}>
            <TextField
              size="small"
              label="Sheet Name"
              value={sheetFilter}
              onChange={(e) => {
                setSheetFilter(e.target.value);
                setPage(0);
              }}
              sx={{ width: 200 }}
              placeholder="Search sheets..."
            />
            <TextField
              select
              size="small"
              label="Domain"
              value={domainFilter}
              onChange={(e) => {
                setDomainFilter(e.target.value);
                setPage(0);
              }}
              sx={{ width: 160 }}
            >
              <MenuItem value="all">All domains</MenuItem>
              {DOMAINS.map((d) => (
                <MenuItem key={d} value={d} sx={{ textTransform: 'capitalize' }}>
                  {d}
                </MenuItem>
              ))}
            </TextField>
            <TextField
              select
              size="small"
              label="Task"
              value={taskFilter}
              onChange={(e) => {
                setTaskFilter(e.target.value);
                setPage(0);
              }}
              sx={{ width: 160 }}
            >
              <MenuItem value="all">All tasks</MenuItem>
              <MenuItem value="matcher">Matcher</MenuItem>
              <MenuItem value="classifier">Classifier</MenuItem>
              <MenuItem value="pipeline">Pipeline</MenuItem>
            </TextField>
          </Stack>
        </Card>

        <StyledTableContainer>
          <Table aria-label="jobs table">
            <StyledTableHead>
              <TableRow>
                <StyledHeaderCell width="15%">Job ID</StyledHeaderCell>
                <StyledHeaderCell width="25%">Sheet Name</StyledHeaderCell>
                <StyledHeaderCell width="15%">Domain</StyledHeaderCell>
                <StyledHeaderCell width="15%">Task</StyledHeaderCell>
                <StyledHeaderCell align="right" width="10%">
                  Total SKUs
                </StyledHeaderCell>
                <StyledHeaderCell align="right" width="10%">
                  High Conf
                </StyledHeaderCell>
                <StyledHeaderCell align="right" width="10%">
                  Details
                </StyledHeaderCell>
              </TableRow>
            </StyledTableHead>
            <StyledTableBody>
              {isLoadingJobs ? (
                <TableSkeleton columns={7} rows={5} />
              ) : paginatedJobs.length === 0 ? (
                <StyledTableRow>
                  <TableCell colSpan={7} align="center">
                    <Typography variant="body2" color="text.secondary">
                      No jobs found.
                    </Typography>
                  </TableCell>
                </StyledTableRow>
              ) : (
                paginatedJobs.map((row) => (
                  <StyledTableRow key={row.id} hover onClick={() => setSelectedJob(row)} sx={{ cursor: 'pointer' }}>
                    <TableCell>{row.id}</TableCell>
                    <TableCell>{row.target_sheet || row.sheet_name || 'N/A'}</TableCell>
                    <TableCell sx={{ textTransform: 'capitalize' }}>{row.domain}</TableCell>
                    <TableCell sx={{ textTransform: 'capitalize' }}>{row.task || row.type}</TableCell>
                    <TableCell align="right">{row.total_items ?? '-'}</TableCell>
                    <TableCell align="right">{row.high_conf ?? '-'}</TableCell>
                    <TableCell align="right" onClick={(e) => e.stopPropagation()}>
                      <IconButton
                        size="small"
                        aria-label="View job details"
                        sx={{ color: 'text.secondary' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedJob(row);
                        }}
                      >
                        <Info size={18} />
                      </IconButton>
                    </TableCell>
                  </StyledTableRow>
                ))
              )}
            </StyledTableBody>
          </Table>
          <TablePagination
            rowsPerPageOptions={[10, 25, 50]}
            component="div"
            count={jobsList.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={(e, newPage) => setPage(newPage)}
            onRowsPerPageChange={(e) => {
              setRowsPerPage(parseInt(e.target.value, 10));
              setPage(0);
            }}
            sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
          />
        </StyledTableContainer>
      </PageContainer>
    );
  }

  // -------------------------
  // SKUS VIEW (FOR SELECTED JOB)
  // -------------------------
  const colSpan = selectedJob.domain === 'all' ? 6 : 5;
  const isClassifier = selectedJob.task?.toLowerCase() === 'classifier';
  const isMatcher = selectedJob.task?.toLowerCase() === 'matcher';

  return (
    <PageContainer>
      <Button startIcon={<ArrowLeft size={16} />} onClick={() => setSelectedJob(null)} sx={{ mb: 2 }} color="inherit">
        Back to SKU Results
      </Button>
      <PageHeader
        title={selectedJob.target_sheet || selectedJob.sheet_name || `Job ${selectedJob.id}`}
        subtitle={`Domain: ${selectedJob.domain} | Total SKUs: ${selectedJob.total_items}`}
        actions={
          <>
            <Button
              variant="outlined"
              color="inherit"
              startIcon={<Download size={15} />}
              endIcon={<ChevronDown size={14} />}
              onClick={(e) => setExportAnchorEl(e.currentTarget)}
              aria-controls={exportAnchorEl ? 'export-csv-menu' : undefined}
              aria-haspopup="true"
              aria-expanded={exportAnchorEl ? 'true' : undefined}
            >
              Download CSV
            </Button>
            <Menu
              id="export-csv-menu"
              anchorEl={exportAnchorEl}
              open={Boolean(exportAnchorEl)}
              onClose={() => setExportAnchorEl(null)}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
              transformOrigin={{ vertical: 'top', horizontal: 'right' }}
              slotProps={{
                paper: {
                  sx: { minWidth: 280, p: 0.5, mt: 0.5, boxShadow: 3 },
                },
              }}
            >
              <MenuItem
                onClick={() => {
                  setExportAnchorEl(null);
                  exportRows('clean');
                }}
                sx={{ py: 1, borderRadius: 1 }}
              >
                <ListItemIcon sx={{ minWidth: 34, color: 'primary.main' }}>
                  <FileSpreadsheet size={18} />
                </ListItemIcon>
                <ListItemText
                  primary="Clean Version (CSV)"
                  secondary="Table columns only (for Google Sheets)"
                  primaryTypographyProps={{ variant: 'body2', fontWeight: 600 }}
                  secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                />
              </MenuItem>
              <Divider sx={{ my: 0.5 }} />
              <MenuItem
                onClick={() => {
                  setExportAnchorEl(null);
                  exportRows('audit');
                }}
                sx={{ py: 1, borderRadius: 1 }}
              >
                <ListItemIcon sx={{ minWidth: 34, color: 'text.secondary' }}>
                  <ShieldCheck size={18} />
                </ListItemIcon>
                <ListItemText
                  primary="Audit Version (CSV)"
                  secondary="Full metadata, scores & confidence"
                  primaryTypographyProps={{ variant: 'body2', fontWeight: 600 }}
                  secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                />
              </MenuItem>
            </Menu>
          </>
        }
      />

      <Card sx={{ p: 2, mb: 3 }}>
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          spacing={2}
          alignItems={{ md: 'center' }}
          flexWrap="wrap"
          useFlexGap
        >
          <TextField
            size="small"
            label="Search SKU Name"
            value={skuSearch}
            onChange={(e) => {
              setSkuSearch(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 220 }}
            placeholder="Search SKU name..."
          />
          <TextField
            size="small"
            label="Basic Type (BT)"
            value={skuBtFilter}
            onChange={(e) => {
              setSkuBtFilter(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 180 }}
            placeholder="Filter BT..."
          />
          <TextField
            size="small"
            label="Generic Keywords (GK)"
            value={skuGkFilter}
            onChange={(e) => {
              setSkuGkFilter(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 180 }}
            placeholder="Filter GK..."
          />
          <TextField
            select
            size="small"
            label="Source"
            value={skuSourceFilter}
            onChange={(e) => {
              setSkuSourceFilter(e.target.value);
              setSkuPage(0);
            }}
            sx={{ width: 160 }}
          >
            <MenuItem value="all">All Sources</MenuItem>
            <MenuItem value="matcher">Matcher</MenuItem>
            <MenuItem value="classifier">Classifier</MenuItem>
          </TextField>
        </Stack>
      </Card>

      <StyledTableContainer>
        <Table aria-label="processed skus table">
          <StyledTableHead>
            <TableRow>
              <StyledHeaderCell>SKU Name</StyledHeaderCell>
              {selectedJob.domain === 'market' && <StyledHeaderCell width="15%">Categories</StyledHeaderCell>}
              <StyledHeaderCell width="45%">Generic Keywords</StyledHeaderCell>
              <StyledHeaderCell width="15%">Basic Type</StyledHeaderCell>
              {selectedJob.domain === 'food' && <StyledHeaderCell width="12%">Region</StyledHeaderCell>}
            </TableRow>
          </StyledTableHead>
          <StyledTableBody>
            {isLoadingSkus ? (
              <TableSkeleton columns={colSpan} rows={5} />
            ) : skusList.length === 0 ? (
              <StyledTableRow>
                <TableCell colSpan={colSpan} align="center">
                  <Typography variant="body2" color="text.secondary">
                    No SKUs found.
                  </Typography>
                </TableCell>
              </StyledTableRow>
            ) : (
              paginatedSkus.map((row) => {
                return (
                  <StyledTableRow
                    key={row.id}
                    hover
                    onClick={() => {
                      setSelectedSku(row);
                    }}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>{row.sku_name}</TableCell>
                    {selectedJob.domain === 'market' && <TableCell>{row.categories || row.region}</TableCell>}
                    <TableCell>{row._formattedGk || formatGk(row.generic_keywords || row.gk_json)}</TableCell>
                    <TableCell>{row.basic_type || row.bt}</TableCell>
                    {selectedJob.domain === 'food' && <TableCell>{row.region}</TableCell>}
                  </StyledTableRow>
                );
              })
            )}
          </StyledTableBody>
        </Table>

        <TablePagination
          rowsPerPageOptions={[100, 250, 500, 800]}
          component="div"
          count={skusList.length}
          rowsPerPage={skuRowsPerPage}
          page={skuPage}
          onPageChange={(e, newPage) => setSkuPage(newPage)}
          onRowsPerPageChange={(e) => {
            setSkuRowsPerPage(parseInt(e.target.value, 10));
            setSkuPage(0);
          }}
          sx={{ borderTop: (theme) => `1px solid ${theme.palette.divider}` }}
        />
      </StyledTableContainer>

      <SideDrawer open={!!selectedSku} onClose={() => setSelectedSku(null)} title={selectedSku?.sku_name}>
        {selectedSku && (
          <Box>
            <Typography variant="caption" color="text.secondary">
              {selectedSku.id} · {selectedSku.batch_id}
            </Typography>
            <Divider sx={{ my: 2 }} />
            {[
              ['Domain', selectedSku.domain],
              ['Generic Keywords', formatGk(selectedSku.generic_keywords || selectedSku.gk_json)],
              ['Basic Type', selectedSku.basic_type || selectedSku.bt],
              ...(selectedSku.domain === 'food'
                ? [['Region', selectedSku.region]]
                : [['Categories', selectedSku.categories || selectedSku.region]]),
              ['Processed at', fmtTime(selectedSku.created_at)],
            ]
              .filter(([, v]) => v !== null && v !== undefined)
              .map(([k, v]) => (
                <Box key={k} sx={{ display: 'flex', justifyContent: 'space-between', py: 0.7 }}>
                  <Typography variant="body2" color="text.secondary">
                    {k}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, textAlign: 'right', maxWidth: '70%' }}>
                    {v}
                  </Typography>
                </Box>
              ))}

            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              Audit Details
            </Typography>
            {[
              ['Source', selectedSku.match_source || '—'],

              // Only for Matcher & Pipeline
              ...(!isClassifier
                ? [
                  ['Matched Catalog Name', selectedSku.matched_catalog_name || '—'],
                  ['Logic Notes', selectedSku.logic_notes || '—'],
                  [
                    'Match Score',
                    selectedSku.match_score != null
                      ? `${Math.round(
                        Number(selectedSku.match_score) > 1
                          ? selectedSku.match_score
                          : selectedSku.match_score * 100
                      )}%`
                      : '—',
                  ],
                ]
                : []),

              // Only for Classifier & Pipeline
              ...(!isMatcher
                ? [
                  [
                    'BT Confidence',
                    selectedSku.bt_confidence != null
                      ? `${Math.round(
                        Number(selectedSku.bt_confidence) > 1
                          ? selectedSku.bt_confidence
                          : selectedSku.bt_confidence * 100
                      )}%`
                      : '—',
                  ],
                  [
                    'GK Confidence',
                    selectedSku.gk_confidence != null
                      ? `${Math.round(
                        Number(selectedSku.gk_confidence) > 1
                          ? selectedSku.gk_confidence
                          : selectedSku.gk_confidence * 100
                      )}%`
                      : '—',
                  ],
                  [
                    'Region/Cat Confidence',
                    selectedSku.region_confidence != null
                      ? `${Math.round(
                        Number(selectedSku.region_confidence) > 1
                          ? selectedSku.region_confidence
                          : selectedSku.region_confidence * 100
                      )}%`
                      : '—',
                  ],
                  [
                    'Final Confidence',
                    selectedSku.confidence != null
                      ? `${Math.round(
                        Number(selectedSku.confidence) > 1 ? selectedSku.confidence : selectedSku.confidence * 100
                      )}%`
                      : '—',
                  ],
                ]
                : []),
            ]
              .filter(([, v]) => v !== null && v !== undefined)
              .map(([k, v]) => (
                <Box key={k} sx={{ display: 'flex', justifyContent: 'space-between', py: 0.7 }}>
                  <Typography variant="body2" color="text.secondary">
                    {k}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500, textAlign: 'right', maxWidth: '70%' }}>
                    {v}
                  </Typography>
                </Box>
              ))}
            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              Rules applied
            </Typography>
            <RulesAppliedList rules={selectedSku.rules_applied_json} />
          </Box>
        )}
      </SideDrawer>

      <ConfirmDialog
        open={Boolean(rerunJob)}
        onClose={() => setRerunJob(null)}
        title="Job Not Completed Fully"
        message={
          rerunJob
            ? `Job ${rerunJob.id} (${rerunJob.target_sheet || rerunJob.sheet_name || 'N/A'}) had a status of '${rerunJob.status
            }' and was not processed fully. Would you like to rerun it?`
            : ''
        }
        onConfirm={async () => {
          if (!rerunJob) return;
          try {
            const res = await retryJob(rerunJob.id);
            enqueueSnackbar(res.message || 'Job rerun successfully initiated!', { variant: 'success' });
            setRerunJob(null);
            navigate('/jobs');
          } catch (err) {
            enqueueSnackbar(err.response?.data?.detail || 'Failed to rerun the job', { variant: 'error' });
            setRerunJob(null);
          }
        }}
        confirmText="Yes, Rerun"
        cancelText="No, Close"
      />
    </PageContainer>
  );
}
