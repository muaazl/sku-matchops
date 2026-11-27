import React from 'react';
import PropTypes from 'prop-types';
import { Divider, List, ListItem, ListItemText } from '@mui/material';
import prettyBytes from 'pretty-bytes';
import { SideDrawer } from '../common/SideDrawer';

export function CollectionDetailsDrawer({ collectionDetails, onClose }) {
  return (
    <SideDrawer open={!!collectionDetails} onClose={onClose} title={collectionDetails?.name || ''} width={400}>
      {collectionDetails && (
        <List>
          <ListItem disableGutters>
            <ListItemText primary="Status" secondary={collectionDetails.status} />
          </ListItem>
          <Divider />
          <ListItem disableGutters>
            <ListItemText primary="Points (approx)" secondary={collectionDetails.points_count?.toLocaleString()} />
          </ListItem>
          <Divider />
          <ListItem disableGutters>
            <ListItemText primary="Segments" secondary={collectionDetails.segments_count} />
          </ListItem>
          <Divider />
          <ListItem disableGutters>
            <ListItemText primary="Vectors configuration" secondary={collectionDetails.vectors_config} />
          </ListItem>
          <Divider />
          <ListItem disableGutters>
            <ListItemText primary="RAM usage" secondary={prettyBytes(collectionDetails.ram_bytes || 0)} />
          </ListItem>
          <Divider />
          <ListItem disableGutters>
            <ListItemText primary="Disk usage" secondary={prettyBytes(collectionDetails.disk_bytes || 0)} />
          </ListItem>
        </List>
      )}
    </SideDrawer>
  );
}

CollectionDetailsDrawer.propTypes = {
  collectionDetails: PropTypes.object,
  onClose: PropTypes.func.isRequired,
};
