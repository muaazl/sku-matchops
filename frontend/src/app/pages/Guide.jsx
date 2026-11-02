import React from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Divider,
  Grid,
  Stack,
  Table,
  TableCell,
  TableRow,
  Typography,
} from '@mui/material';
import { ArrowRight, Tag, Gauge, Compass, Store, UtensilsCrossed, ShoppingBasket } from 'lucide-react';
import { PageContainer, PageHeader } from '../components/ui';
import {
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledHeaderCell,
  StyledTableRow,
} from '../../components/Common/StyledTable';

const TAG_EXAMPLES = [
  { sku: 'Egg Roll', gk: 'Short Eats, Rolls, Egg Roll', bt: 'Rolls', region: 'Sri Lankan' },
  { sku: 'Chicken Pizza', gk: 'Pizza, Chicken Pizza', bt: 'Pizza', region: 'Italian' },
  { sku: 'French Fries', gk: 'Fast Food, Fries', bt: 'Fries', region: 'Western' },
  { sku: 'Cappuccino', gk: 'Beverage, Coffee, Cappuccino', bt: 'Cappuccino', region: 'Sri Lankan' },
  { sku: 'Chicken Fried Rice', gk: 'Fried Rice, Chicken Fried Rice', bt: 'Fried Rice', region: 'Sri Lankan Chinese' },
  {
    sku: 'Sandwich Bread',
    gk: 'Sandwich Bread, White Sandwich Bread, Bakery',
    bt: 'Sandwich Bread',
    region: 'Sri Lankan',
  },
];

const STATUS_ROWS = [
  {
    status: 'AUTO',
    range: '≥ 80%',
    color: 'success',
    meaning: 'Tag is trusted as-is and flows straight through the pipeline.',
  },
  {
    status: 'REVIEW',
    range: '50% – 79%',
    color: 'warning',
    meaning: 'Tag is kept but flagged — worth a second look before treating it as ground truth.',
  },
  {
    status: 'LOW',
    range: '< 50%',
    color: 'error',
    meaning: 'Tag is unreliable; the SKU is escalated rather than auto-approved.',
  },
];

const SOURCE_GLOSSARY = [
  {
    source: 'trained',
    meaning: 'Predicted by a classifier trained on labeled catalog data — the default, highest-volume path.',
  },
  {
    source: 'zero-shot',
    meaning: 'Predicted by embedding similarity when no trained classifier confidently covers the SKU.',
  },
  { source: 'override', meaning: 'Forced by a Rules Engine rule — a human decided this SKU always gets this tag.' },
  { source: 'keyword', meaning: 'Matched directly off a keyword in the SKU name (fast, deterministic fallback).' },
  { source: 'synthetic', meaning: 'Injected automatically, e.g. combining brand + category into a Generic Keyword.' },
  {
    source: 'conflict',
    meaning: 'Two signals disagreed on the tag — surfaced so it can be resolved rather than silently picked.',
  },
];

const OUTLET_EXAMPLES = [
  { outlet: 'Pizza Hut', region: 'Italian' },
  { outlet: 'Keells', region: 'Bakery' },
  { outlet: 'Chola Authentic Indian Restaurant', region: 'South Indian / North Indian' },
  { outlet: 'Hotel De Plaza', region: 'Sri Lankan Kottu' },
];

function SectionCard({ icon: Icon, title, subheader, children }) {
  return (
    <Card>
      <CardHeader
        avatar={<Icon size={20} />}
        title={title}
        subheader={subheader}
        titleTypographyProps={{ variant: 'subtitle1', fontWeight: 600 }}
      />
      <CardContent sx={{ pt: 0 }}>{children}</CardContent>
    </Card>
  );
}

function FlowChip({ label, sub }) {
  return (
    <Box sx={{ textAlign: 'center' }}>
      <Chip label={label} sx={{ fontWeight: 500 }} />
      {sub && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          {sub}
        </Typography>
      )}
    </Box>
  );
}

export default function Guide() {
  return (
    <PageContainer>
      <PageHeader
        title="Guide"
        subtitle="How SKU tagging works in this system — for anyone matching, auditing, or reviewing SKUs."
      />

      <Stack spacing={3}>
        <SectionCard
          icon={Tag}
          title="How a SKU gets tagged"
          subheader="Every processed SKU gets three tags: Basic Type, Generic Keyword, and Region (or Category for market SKUs)"
        >
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1.5}
            alignItems="center"
            justifyContent="center"
            sx={{ py: 2, flexWrap: 'wrap', rowGap: 2 }}
          >
            <FlowChip label='"Egg Roll"' sub="SKU name" />
            <ArrowRight size={18} style={{ opacity: 0.5 }} />
            <FlowChip label="Short Eats → Rolls → Egg Roll" sub="Generic Keyword (broad → specific)" />
            <ArrowRight size={18} style={{ opacity: 0.5 }} />
            <FlowChip label="Rolls" sub="Basic Type" />
            <ArrowRight size={18} style={{ opacity: 0.5 }} />
            <FlowChip label="Sri Lankan" sub="Region" />
          </Stack>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            The <strong>Generic Keyword</strong> stack always runs from the broadest bucket down to the most specific
            name — that ordering is what lets search and matching work at any level of granularity. The{' '}
            <strong>Basic Type</strong> is the single canonical name for the dish/product family, and{' '}
            <strong>Region</strong> is its cuisine origin. A few more worked examples across categories:
          </Typography>

          <StyledTableContainer>
            <Table aria-label="tagging examples" size="small">
              <StyledTableHead>
                <TableRow>
                  <StyledHeaderCell>SKU Name</StyledHeaderCell>
                  <StyledHeaderCell>Generic Keyword</StyledHeaderCell>
                  <StyledHeaderCell>Basic Type</StyledHeaderCell>
                  <StyledHeaderCell>Region</StyledHeaderCell>
                </TableRow>
              </StyledTableHead>
              <StyledTableBody>
                {TAG_EXAMPLES.map((row) => (
                  <StyledTableRow key={row.sku}>
                    <TableCell sx={{ fontWeight: 500 }}>{row.sku}</TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {row.gk}
                      </Typography>
                    </TableCell>
                    <TableCell>{row.bt}</TableCell>
                    <TableCell>
                      <Chip size="small" label={row.region} variant="outlined" />
                    </TableCell>
                  </StyledTableRow>
                ))}
              </StyledTableBody>
            </Table>
          </StyledTableContainer>
        </SectionCard>

        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              icon={Gauge}
              title="Confidence & status"
              subheader="What AUTO / REVIEW / LOW mean for a tag you see in the Dashboard or Interactive tools"
            >
              <StyledTableContainer>
                <Table aria-label="confidence status" size="small">
                  <StyledTableHead>
                    <TableRow>
                      <StyledHeaderCell>Status</StyledHeaderCell>
                      <StyledHeaderCell>Confidence</StyledHeaderCell>
                      <StyledHeaderCell>Meaning</StyledHeaderCell>
                    </TableRow>
                  </StyledTableHead>
                  <StyledTableBody>
                    {STATUS_ROWS.map((row) => (
                      <StyledTableRow key={row.status}>
                        <TableCell>
                          <Chip size="small" label={row.status} color={row.color} />
                        </TableCell>
                        <TableCell sx={{ whiteSpace: 'nowrap' }}>{row.range}</TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {row.meaning}
                          </Typography>
                        </TableCell>
                      </StyledTableRow>
                    ))}
                  </StyledTableBody>
                </Table>
              </StyledTableContainer>
            </SectionCard>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              icon={Compass}
              title="Where a tag comes from"
              subheader="The source attached to every predicted tag"
            >
              <Stack spacing={1.25}>
                {SOURCE_GLOSSARY.map((row) => (
                  <Box key={row.source}>
                    <Stack direction="row" spacing={1} alignItems="baseline">
                      <Chip size="small" label={row.source} variant="outlined" sx={{ fontFamily: 'monospace' }} />
                    </Stack>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                      {row.meaning}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </SectionCard>
          </Grid>
        </Grid>

        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard icon={UtensilsCrossed} title="Food domain" subheader="Basic Type + Generic Keyword + Region">
              <Typography variant="body2" color="text.secondary">
                Food SKUs are tagged by cuisine — e.g. a <em>Chicken Kottu</em> gets Basic Type <strong>Kottu</strong>{' '}
                and Region <strong>Sri Lankan</strong>. Region answers "where does this dish come from," which is what
                powers cuisine-based search and filtering.
              </Typography>
            </SectionCard>
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              icon={ShoppingBasket}
              title="Market domain"
              subheader="Basic Type + Generic Keyword + Category"
            >
              <Typography variant="body2" color="text.secondary">
                Market (grocery/retail) SKUs replace Region with <strong>Category</strong> — e.g. <strong>Dairy</strong>
                , <strong>Snacks</strong>, <strong>Beverages</strong> — since retail products aren't defined by cuisine
                of origin, but by the shelf/aisle they'd sit on.
              </Typography>
            </SectionCard>
          </Grid>
        </Grid>

        <SectionCard
          icon={Store}
          title="Outlet-level cuisine tagging"
          subheader="Sometimes the Region tag comes from the restaurant brand itself, not just the SKU name"
        >
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            A generic name like "Chicken Fried Rice" doesn't say much about cuisine on its own — but if it's sold at an
            outlet that's already known to be, say, an Italian restaurant, that context can drive the Region tag
            directly.
          </Typography>
          <StyledTableContainer>
            <Table aria-label="outlet examples" size="small">
              <StyledTableHead>
                <TableRow>
                  <StyledHeaderCell>Outlet</StyledHeaderCell>
                  <StyledHeaderCell>Inferred Region</StyledHeaderCell>
                </TableRow>
              </StyledTableHead>
              <StyledTableBody>
                {OUTLET_EXAMPLES.map((row) => (
                  <StyledTableRow key={row.outlet}>
                    <TableCell sx={{ fontWeight: 500 }}>{row.outlet}</TableCell>
                    <TableCell>
                      <Chip size="small" label={row.region} variant="outlined" />
                    </TableCell>
                  </StyledTableRow>
                ))}
              </StyledTableBody>
            </Table>
          </StyledTableContainer>
        </SectionCard>

        <Divider sx={{ my: 1 }} />
        <Typography variant="caption" color="text.secondary">
          Want to see this data live? The Dashboard's "Top Basic Types" and "Region / Category distribution" charts
          reflect the actual tags currently in the catalog.
        </Typography>
      </Stack>
    </PageContainer>
  );
}
