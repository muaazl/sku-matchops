import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Button, Chip, Grid, MenuItem, Stack, TextField, ToggleButton, ToggleButtonGroup } from '@mui/material';
import {
  Bolt,
  Cancel,
  CheckCircle,
  Label,
  MenuBook,
  PieChart,
  Psychology,
  Public,
  Timeline,
  TrendingUp,
} from '@mui/icons-material';
import { BasicTypeBarChart } from '../components/charts/BasicTypeBarChart';
import { ConfidenceDoughnutChart } from '../components/charts/ConfidenceDoughnutChart';
import { MatchSourcePieChart } from '../components/charts/MatchSourcePieChart';
import { RegionDistributionChart } from '../components/charts/RegionDistributionChart';
import { VolumeTrendChart } from '../components/charts/VolumeTrendChart';
import { EmptyState } from '../components/common/EmptyState';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { SectionCard } from '../components/common/SectionCard';
import { StatCard } from '../components/common/StatCard';
import { useDashboardStats } from '../hooks/useDashboardStats';

export default function Dashboard() {
  const navigate = useNavigate();
  const [domain, setDomain] = useState('all');
  const [timeframe, setTimeframe] = useState('30d');

  const {
    stats,
    confidenceDistribution,
    matchSourceDistribution,
    domainBreakdown,
    volumeTrend,
    basicTypeDistribution,
    regionDistribution,
    statsLoading,
    tagLoading,
  } = useDashboardStats(domain, timeframe);

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
                  ? `Food: ${domainBreakdown.food.count.toLocaleString()} · ` +
                    `Market: ${domainBreakdown.market.count.toLocaleString()}`
                  : 'Across active catalog'
              }
              color="info"
              isLoading={statsLoading}
            />
          </Grid>
        </Grid>

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
