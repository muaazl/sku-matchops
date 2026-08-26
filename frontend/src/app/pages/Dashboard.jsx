import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  LinearProgress,
  TextField,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
} from '@mui/material';
import {
  CheckCircle2,
  XCircle,
  Activity,
  Upload,
  Search as SearchIcon,
  PlaySquare,
  SlidersHorizontal,
  Zap,
} from 'lucide-react';
import { PageContainer, PageHeader } from '../components/ui';
import { ConfidenceDoughnutChart, MatchSourcePieChart, VolumeTrendChart } from '../components/Charts';
import { useQuery } from '@tanstack/react-query';
import { getDashboardStats } from '../api';

function StatCard({ icon: Icon, label, value, subtext, color }) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Box
          sx={{
            width: 44,
            height: 44,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: (t) => t.palette[color].main + '1f',
            color: (t) => t.palette[color].main,
            flexShrink: 0,
          }}
        >
          <Icon size={22} />
        </Box>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600, lineHeight: 1.1 }}>
            {value}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {label}
          </Typography>
          {subtext && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
              {subtext}
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();

  // Operational Filters State (Defaulting to 30d Month view)
  const [domain, setDomain] = useState('all');
  const [timeframe, setTimeframe] = useState('30d');

  // Query 1: Dashboard stats from sqlite backend API
  const { data: dashboardData, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard-stats', domain, timeframe],
    queryFn: () => getDashboardStats({ domain, timeframe }),
    staleTime: 15 * 1000,
  });

  const stats = dashboardData?.stats || {
    totalProcessedSkus: 0,
    avgConfidencePct: 0.0,
    highConfidenceCount: 0,
    highConfidencePct: 0.0,
    mediumConfidenceCount: 0,
    mediumConfidencePct: 0.0,
    lowConfidenceCount: 0,
    lowConfidencePct: 0.0,
  };

  const confidenceDistribution = dashboardData?.confidenceDistribution || [];
  const matchSourceDistribution = dashboardData?.matchSourceDistribution || [];
  const domainBreakdown = dashboardData?.domainBreakdown || {};
  const volumeTrend = dashboardData?.volumeTrend || [];

  const isLoading = statsLoading;

  return (
    <PageContainer>
      <PageHeader
        title="Dashboard"
        subtitle="Job health, matching accuracy, confidence distribution, and vector store stats."
        actions={
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
            <ToggleButtonGroup size="small" value={domain} exclusive onChange={(_, val) => val && setDomain(val)}>
              <ToggleButton value="all" sx={{ px: 1.5, textTransform: 'capitalize' }}>
                All Domain
              </ToggleButton>
              <ToggleButton value="food" sx={{ px: 1.5, textTransform: 'capitalize' }}>
                Food
              </ToggleButton>
              <ToggleButton value="market" sx={{ px: 1.5, textTransform: 'capitalize' }}>
                Market
              </ToggleButton>
            </ToggleButtonGroup>

            <TextField
              select
              size="small"
              label="Timeframe"
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              sx={{ width: 150 }}
            >
              <MenuItem value="24h">Last 24 hours</MenuItem>
              <MenuItem value="7d">Last 7 days</MenuItem>
              <MenuItem value="30d">Last 30 days</MenuItem>
              <MenuItem value="all">All time</MenuItem>
            </TextField>
          </Box>
        }
      />

      {isLoading && <LinearProgress sx={{ mb: 3, borderRadius: 2 }} />}

      {/* Quick Links */}
      <Typography variant="h6" sx={{ mb: 2 }}>
        Quick Links
      </Typography>
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {[
          { title: 'Process SKUs', icon: Upload, path: '/process-skus', color: 'primary' },
          { title: 'Catalog Search', icon: SearchIcon, path: '/catalog', color: 'info' },
          { title: 'Interactive', icon: PlaySquare, path: '/interactive', color: 'success' },
          { title: 'Rules Engine', icon: SlidersHorizontal, path: '/rules', color: 'warning' },
        ].map((link) => (
          <Grid size={{ xs: 12, sm: 6, md: 3 }} key={link.title}>
            <Card
              sx={{ cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }}
              onClick={() => navigate(link.path)}
            >
              <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2, '&:last-child': { pb: 2 } }}>
                <Box sx={{ color: (t) => t.palette[link.color].main, display: 'flex' }}>
                  <link.icon size={24} />
                </Box>
                <Typography variant="subtitle1" fontWeight={500}>
                  {link.title}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* KPI Stat Cards */}
      <Grid container spacing={3} sx={{ mb: 1 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            icon={Zap}
            label="Avg match confidence"
            value={`${stats.avgConfidencePct}%`}
            subtext="Target ≥ 85.0%"
            color="primary"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            icon={CheckCircle2}
            label="Auto-approved (≥85%)"
            value={stats.highConfidenceCount.toLocaleString()}
            subtext={`${stats.highConfidencePct}% of total`}
            color="success"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            icon={XCircle}
            label="Escalated (<60%)"
            value={stats.lowConfidenceCount.toLocaleString()}
            subtext={`${stats.lowConfidencePct}% flagged for audit`}
            color="error"
          />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <StatCard
            icon={Activity}
            label="Total SKUs processed"
            value={stats.totalProcessedSkus.toLocaleString()}
            subtext={
              domainBreakdown.food && domainBreakdown.market
                ? `Food: ${domainBreakdown.food.count.toLocaleString()} · Market: ${domainBreakdown.market.count.toLocaleString()}`
                : 'Across active catalog'
            }
            color="info"
          />
        </Grid>
      </Grid>

      {/* Visualizations Grid */}
      <Grid container spacing={3} sx={{ mt: 0 }}>
        {/* Match Quality & Confidence Distribution */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%' }}>
            <CardHeader
              title="Match quality & confidence distribution"
              titleTypographyProps={{ variant: 'subtitle1' }}
            />
            <CardContent>
              <Box sx={{ height: 280 }}>
                {confidenceDistribution.some((c) => c.count > 0) ? (
                  <ConfidenceDoughnutChart data={confidenceDistribution} />
                ) : (
                  <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      No SKU match data available
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Match Algorithm Attribution */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%' }}>
            <CardHeader title="Matching algorithm attribution" titleTypographyProps={{ variant: 'subtitle1' }} />
            <CardContent>
              <Box sx={{ height: 280 }}>
                {matchSourceDistribution.length > 0 ? (
                  <MatchSourcePieChart data={matchSourceDistribution} />
                ) : (
                  <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      No pipeline source data available
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* SKU Volume & Confidence Trend (Max Width) */}
        <Grid size={12}>
          <Card>
            <CardHeader
              title="SKU volume & confidence trend"
              titleTypographyProps={{ variant: 'subtitle1' }}
              action={<Chip size="small" label={timeframe === '24h' ? 'hourly' : 'daily'} variant="outlined" />}
            />
            <CardContent>
              <Box sx={{ height: 300 }}>
                {volumeTrend.length > 0 ? (
                  <VolumeTrendChart data={volumeTrend} />
                ) : (
                  <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      No volume history available
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </PageContainer>
  );
}
