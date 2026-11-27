import React, { useState } from 'react';
import { CollectionDetailsDrawer } from '../components/collections/CollectionDetailsDrawer';
import { CollectionsTable } from '../components/collections/CollectionsTable';
import { PointPayloadDrawer } from '../components/collections/PointPayloadDrawer';
import { VectorSearchPanel } from '../components/collections/VectorSearchPanel';
import { PageContainer } from '../components/common/PageContainer';
import { PageHeader } from '../components/common/PageHeader';
import { useVectorCollections } from '../hooks/useVectorCollections';

export default function Collections() {
  const [detailsDrawer, setDetailsDrawer] = useState(null);
  const [payloadDrawer, setPayloadDrawer] = useState(null);

  const {
    collections,
    isLoading,
    collection,
    setCollection,
    query,
    setQuery,
    searchMode,
    setSearchMode,
    matcherDomain,
    results,
    isSearching,
    runSearch,
  } = useVectorCollections();

  return (
    <PageContainer>
      <PageHeader
        title="Collections"
        subtitle="Qdrant collections backing the matcher, with live stats and a vector search box."
      />

      <CollectionsTable collections={collections} isLoading={isLoading} onSelectCollection={setDetailsDrawer} />

      <VectorSearchPanel
        collections={collections}
        collection={collection}
        onCollectionChange={setCollection}
        searchMode={searchMode}
        onSearchModeChange={setSearchMode}
        matcherDomain={matcherDomain}
        query={query}
        onQueryChange={setQuery}
        isSearching={isSearching}
        onRunSearch={runSearch}
        results={results}
        onSelectPayload={setPayloadDrawer}
      />

      <CollectionDetailsDrawer collectionDetails={detailsDrawer} onClose={() => setDetailsDrawer(null)} />
      <PointPayloadDrawer payloadPoint={payloadDrawer} onClose={() => setPayloadDrawer(null)} />
    </PageContainer>
  );
}
