import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { Autocomplete, CircularProgress, TextField } from '@mui/material';
import { useTagSuggestions } from '../../hooks/useTagSuggestions';

export default function TagAutocomplete({ value, onChange, type, ruleDomain, placeholder }) {
  const [inputValue, setInputValue] = useState(value || '');
  const { dataset, options, loading } = useTagSuggestions({ type, ruleDomain, inputValue });

  useEffect(() => {
    setInputValue(value || '');
  }, [value]);

  if (type === 'flavor_is') {
    return (
      <Autocomplete
        size="small"
        options={['meat', 'vegetable', 'seafood']}
        value={value || ''}
        onChange={(event, newValue) => {
          onChange(newValue || '');
        }}
        renderInput={(params) => <TextField {...params} size="small" placeholder={placeholder} sx={{ flexGrow: 1 }} />}
        sx={{ flexGrow: 1 }}
      />
    );
  }

  if (!dataset) {
    return (
      <TextField
        size="small"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        sx={{ flexGrow: 1 }}
      />
    );
  }

  return (
    <Autocomplete
      freeSolo
      size="small"
      options={options}
      loading={loading}
      value={value || ''}
      onChange={(event, newValue) => {
        onChange(newValue || '');
      }}
      inputValue={inputValue}
      onInputChange={(event, newInputValue) => {
        setInputValue(newInputValue);
        onChange(newInputValue);
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          placeholder={placeholder}
          sx={{ flexGrow: 1 }}
          InputProps={{
            ...params.InputProps,
            endAdornment: (
              <>
                {loading ? <CircularProgress color="inherit" size={16} /> : null}
                {params.InputProps.endAdornment}
              </>
            ),
          }}
        />
      )}
      sx={{ flexGrow: 1 }}
    />
  );
}

TagAutocomplete.propTypes = {
  value: PropTypes.string,
  onChange: PropTypes.func.isRequired,
  type: PropTypes.string,
  ruleDomain: PropTypes.string,
  placeholder: PropTypes.string,
};
