import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Button, Chip, Grid, MenuItem, Stack, TextField, ToggleButton, ToggleButtonGroup } from '@mui/material';
import {
  Bolt,
  CheckCircle,
  Cancel,
  Timeline,
  PieChart,
  Psychology,
  TrendingUp,
  MenuBook,
  Label,
  Public,
} from '@mui/icons-material';
import { EmptyState, PageContainer, PageHeader, SectionCard, StatCard } from '../components/ui';
import {
  BasicTypeBarChart,
  ConfidenceDoughnutChart,
  MatchSourcePieChart,
  RegionDistributionChart,
  VolumeTrendChart,
} from '../components/Charts';
import { useQuery } from '@tanstack/react-query';
import { getDashboardStats, getTagStats } from '../api';

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
  const rawSources = dashboardData?.matchSourceDistribution || [];
  const matchSourceDistribution = React.useMemo(() => {
    const map = {};
    for (const item of rawSources) {
      const src = (item.source || '').trim().toLowerCase();
      const norm = src === 'classifier' ? 'Classifier' : 'Matcher';
      map[norm] = (map[norm] || 0) + (item.count || 0);
    }
    const total = Object.values(map).reduce((a, b) => a + b, 0);
    return Object.entries(map)
      .filter((entry) => entry[1] > 0)
      .map(([source, count]) => ({
        source,
        count,
        pct: total > 0 ? Math.round((count / total) * 1000) / 10 : 0,
      }));
  }, [rawSources]);
  const domainBreakdown = dashboardData?.domainBreakdown || {};
  const volumeTrend = dashboardData?.volumeTrend || [];

  const basicTypeDistribution = tagData?.basicTypeDistribution || [];
  const regionDistribution = tagData?.regionDistribution || [];

  return (
    <PageContainer>
      <PageHeader
        title="Dashboard"
        subtitle="Job health, matching accuracy, confidence distribution, and vector store stats."
        actions={
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
            <ToggleButtonGroup
              size="small"
              value={domain}
              exclusive
              onChange={(_, val) => val && setDomain(val)}
              aria-label="Filter dashboard by domain"
            >
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
              inputProps={{ 'aria-label': 'Filter dashboard by timeframe' }}
            >
              <MenuItem value="24h">Last 24 hours</MenuItem>
              <MenuItem value="7d">Last 7 days</MenuItem>
              <MenuItem value="30d">Last 30 days</MenuItem>
              <MenuItem value="all">All time</MenuItem>
            </TextField>
          </Box>
        }
      />

      <Stack spacing={3}>
        {/* Row 1: KPI Stat Cards */}
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard
              icon={Bolt}
              label="Avg match confidence"
              value={`${stats.avgConfidencePct}%`}
              subtext="Target ≥ 85.0%"
              color="primary"
              isLoading={statsLoading}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard
              icon={CheckCircle}
              label="Auto-approved (≥85%)"
              value={stats.highConfidenceCount.toLocaleString()}
              subtext={`${stats.highConfidencePct}% of total`}
              color="success"
              isLoading={statsLoading}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard
              icon={Cancel}
              label="Escalated (<60%)"
              value={stats.lowConfidenceCount.toLocaleString()}
              subtext={`${stats.lowConfidencePct}% flagged for audit`}
              color="error"
              isLoading={statsLoading}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <StatCard
              icon={Timeline}
              label="Total SKUs processed"
              value={stats.totalProcessedSkus.toLocaleString()}
              subtext={
                domainBreakdown.food && domainBreakdown.market
                  ? `Food: ${domainBreakdown.food.count.toLocaleString()} · Market: ${domainBreakdown.market.count.toLocaleString()}`
                  : 'Across active catalog'
              }
              color="info"
              isLoading={statsLoading}
            />
          </Grid>
        </Grid>

        {/* Row 2: Match Quality & Confidence Distribution / Matching Algorithm Attribution */}
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              title="Match quality & confidence distribution"
              subheader="Confidence distribution across processed batches"
              isLoading={statsLoading}
              minHeight={280}
            >
              <Box sx={{ height: 280 }}>
                {confidenceDistribution.some((c) => c.count > 0) ? (
                  <ConfidenceDoughnutChart data={confidenceDistribution} />
                ) : (
                  <EmptyState
                    icon={PieChart}
                    title="No SKU match data"
                    description="No confidence score distribution recorded for this domain and timeframe."
                    minHeight={240}
                  />
                )}
              </Box>
            </SectionCard>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              title="Matching algorithm attribution"
              subheader="Distribution between Rule Classifier & Vector Matcher"
              isLoading={statsLoading}
              minHeight={280}
            >
              <Box sx={{ height: 280 }}>
                {matchSourceDistribution.length > 0 ? (
                  <MatchSourcePieChart data={matchSourceDistribution} />
                ) : (
                  <EmptyState
                    icon={Psychology}
                    title="No pipeline source data"
                    description="No algorithm attribution data available for the selected filters."
                    minHeight={240}
                  />
                )}
              </Box>
            </SectionCard>
          </Grid>
        </Grid>

        {/* Row 3: SKU Volume & Confidence Trend (Max Width) */}
        <Grid container spacing={3}>
          <Grid size={12}>
            <SectionCard
              title="SKU volume & confidence trend"
              subheader="Historical throughput and average confidence over time"
              isLoading={statsLoading}
              minHeight={300}
              action={
                <Chip
                  size="small"
                  label={timeframe === '24h' ? 'hourly' : 'daily'}
                  variant="outlined"
                  sx={{ fontWeight: 600, textTransform: 'capitalize' }}
                />
              }
            >
              <Box sx={{ height: 300 }}>
                {volumeTrend.length > 0 ? (
                  <VolumeTrendChart data={volumeTrend} />
                ) : (
                  <EmptyState
                    icon={TrendingUp}
                    title="No volume history"
                    description="No SKU volume trend records found for the selected period."
                    minHeight={260}
                  />
                )}
              </Box>
            </SectionCard>
          </Grid>
        </Grid>

        {/* Row 4: Top Basic Types / Region-Category Distribution */}
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              title="Top Basic Types"
              subheader="What SKUs are most commonly tagged as"
              isLoading={tagLoading}
              minHeight={280}
              action={
                <Button
                  size="small"
                  startIcon={<MenuBook sx={{ fontSize: 16 }} />}
                  onClick={() => navigate('/guide')}
                  sx={{ textTransform: 'none', fontWeight: 600 }}
                >
                  How tagging works
                </Button>
              }
            >
              <Box sx={{ height: 280 }}>
                {basicTypeDistribution.length > 0 ? (
                  <BasicTypeBarChart data={basicTypeDistribution} />
                ) : (
                  <EmptyState
                    icon={Label}
                    title="No Basic Type data"
                    description="No basic type tags identified in the selected dataset."
                    minHeight={240}
                  />
                )}
              </Box>
            </SectionCard>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              title="Region / Category distribution"
              subheader="Where the catalog's cuisine & category tags land"
              isLoading={tagLoading}
              minHeight={280}
            >
              <Box sx={{ height: 280 }}>
                {regionDistribution.length > 0 ? (
                  <RegionDistributionChart data={regionDistribution} />
                ) : (
                  <EmptyState
                    icon={Public}
                    title="No Region / Category data"
                    description="No region or category distribution available for the selected filters."
                    minHeight={240}
                  />
                )}
              </Box>
            </SectionCard>
          </Grid>
        </Grid>
      </Stack>
    </PageContainer>
  );
}
