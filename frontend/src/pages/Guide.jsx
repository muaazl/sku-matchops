import React from 'react';
import PropTypes from 'prop-types';
import { Box, Chip, Divider, Grid, Stack, Table, TableCell, TableRow, Typography } from '@mui/material';
import { ArrowForward, Explore, Label, Restaurant, ShoppingBasket, Speed, Storefront } from '@mui/icons-material';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { SectionCard } from '../components/common/SectionCard';
import {
  StyledHeaderCell,
  StyledTableBody,
  StyledTableContainer,
  StyledTableHead,
  StyledTableRow,
} from '../components/common/StyledTable';
import { OUTLET_EXAMPLES, SOURCE_GLOSSARY, STATUS_ROWS, TAG_EXAMPLES } from '../utils/guideData';

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

FlowChip.propTypes = {
  label: PropTypes.string.isRequired,
  sub: PropTypes.string,
};

export default function Guide() {
  return (
    <PageContainer>
      <PageHeader
        title="Guide"
        subtitle="How SKU tagging works in this system — for anyone matching, auditing, or reviewing SKUs."
      />

      <Stack spacing={3}>
        <SectionCard
          icon={Label}
          title="How a SKU gets tagged"
          subheader={
            'Every processed SKU gets three tags: Basic Type, Generic Keyword, ' +
            'and Region (or Category for market SKUs)'
          }
          contentSx={{ pt: 0 }}
        >
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            spacing={1.5}
            alignItems="center"
            justifyContent="center"
            sx={{ py: 2, flexWrap: 'wrap', rowGap: 2 }}
          >
            <FlowChip label='"Egg Roll"' sub="SKU name" />
            <ArrowForward sx={{ fontSize: 18, opacity: 0.5 }} />
            <FlowChip label="Short Eats → Rolls → Egg Roll" sub="Generic Keyword (broad → specific)" />
            <ArrowForward sx={{ fontSize: 18, opacity: 0.5 }} />
            <FlowChip label="Rolls" sub="Basic Type" />
            <ArrowForward sx={{ fontSize: 18, opacity: 0.5 }} />
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
              icon={Speed}
              title="Confidence & status"
              subheader="What AUTO / REVIEW / LOW mean for a tag you see in the Dashboard or Interactive tools"
              contentSx={{ pt: 0 }}
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
              icon={Explore}
              title="Where a tag comes from"
              subheader="The source attached to every predicted tag"
              contentSx={{ pt: 0 }}
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
            <SectionCard
              icon={Restaurant}
              title="Food domain"
              subheader="Basic Type + Generic Keyword + Region"
              contentSx={{ pt: 0 }}
            >
              <Typography variant="body2" color="text.secondary">
                {'Food SKUs are tagged by cuisine — e.g. a '}
                <em>Chicken Kottu</em>
                {' gets Basic Type '}
                <strong>Kottu</strong>
                {' and Region '}
                <strong>Sri Lankan</strong>
                {'. Region answers "where does this dish come from," which is what powers '}
                {'cuisine-based search and filtering.'}
              </Typography>
            </SectionCard>
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <SectionCard
              icon={ShoppingBasket}
              title="Market domain"
              subheader="Basic Type + Generic Keyword + Category"
              contentSx={{ pt: 0 }}
            >
              <Typography variant="body2" color="text.secondary">
                Market (grocery/retail) SKUs replace Region with <strong>Category</strong> — e.g. <strong>Dairy</strong>
                , <strong>Snacks</strong>, <strong>Beverages</strong> — since retail products aren&apos;t defined by
                cuisine of origin, but by the shelf/aisle they&apos;d sit on.
              </Typography>
            </SectionCard>
          </Grid>
        </Grid>

        <SectionCard
          icon={Storefront}
          title="Outlet-level cuisine tagging"
          subheader="Sometimes the Region tag comes from the restaurant brand itself, not just the SKU name"
          contentSx={{ pt: 0 }}
        >
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            A generic name like &quot;Chicken Fried Rice&quot; doesn&apos;t say much about cuisine on its own — but if
            it&apos;s sold at an outlet that&apos;s already known to be, say, an Italian restaurant, that context can
            drive the Region tag directly.
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
          Want to see this data live? The Dashboard&apos;s &quot;Top Basic Types&quot; and &quot;Region / Category
          distribution&quot; charts reflect the actual tags currently in the catalog.
        </Typography>
      </Stack>
    </PageContainer>
  );
}
