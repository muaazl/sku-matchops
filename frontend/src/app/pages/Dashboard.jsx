import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Stack,
  Typography,
  LinearProgress,
  TextField,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
} from '@mui/material';
import { CheckCircle2, XCircle, Activity, Zap, BookOpen } from 'lucide-react';
import { PageContainer, PageHeader } from '../components/ui';
import {
  ConfidenceDoughnutChart,
  MatchSourcePieChart,
  VolumeTrendChart,
  BasicTypeBarChart,
  RegionDistributionChart,
} from '../components/Charts';
import { useQuery } from '@tanstack/react-query';
import { getDashboardStats, getTagStats } from '../api';

export function StatCard({ icon: Icon, label, value, subtext, color }) {
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

  // Query 2: Basic Type / Region-Category tagging breakdown
  const { data: tagData, isLoading: tagLoading } = useQuery({
    queryKey: ['tag-stats', domain, timeframe],
    queryFn: () => getTagStats({ domain, timeframe }),
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

  const basicTypeDistribution = tagData?.basicTypeDistribution || [];
  const regionDistribution = tagData?.regionDistribution || [];

  const isLoading = statsLoading || tagLoading;

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

      <Stack spacing={3}>
        {/* Row 1: KPI Stat Cards */}
        <Grid container spacing={3}>
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

        {/* Row 2: Match Quality & Confidence Distribution / Matching Algorithm Attribution */}
        <Grid container spacing={3}>
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
        </Grid>

        {/* Row 3: SKU Volume & Confidence Trend (Max Width) */}
        <Grid container spacing={3}>
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

        {/* Row 4: Top Basic Types / Region-Category Distribution */}
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <Card sx={{ height: '100%' }}>
              <CardHeader
                title="Top Basic Types"
                subheader="What SKUs are most commonly tagged as"
                titleTypographyProps={{ variant: 'subtitle1' }}
                action={
                  <Button size="small" startIcon={<BookOpen size={16} />} onClick={() => navigate('/guide')}>
                    How tagging works
                  </Button>
                }
              />
              <CardContent>
                <Box sx={{ height: 280 }}>
                  {basicTypeDistribution.length > 0 ? (
                    <BasicTypeBarChart data={basicTypeDistribution} />
                  ) : (
                    <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        No Basic Type data available
                      </Typography>
                    </Box>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <Card sx={{ height: '100%' }}>
              <CardHeader
                title="Region / Category distribution"
                subheader="Where the catalog's cuisine & category tags land"
                titleTypographyProps={{ variant: 'subtitle1' }}
              />
              <CardContent>
                <Box sx={{ height: 280 }}>
                  {regionDistribution.length > 0 ? (
                    <RegionDistributionChart data={regionDistribution} />
                  ) : (
                    <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography variant="body2" color="text.secondary">
                        No Region / Category data available
                      </Typography>
                    </Box>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Stack>
    </PageContainer>
  );
}
