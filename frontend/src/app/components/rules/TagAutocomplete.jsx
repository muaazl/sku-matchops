import React, { useState, useEffect, useMemo } from 'react';
import PropTypes from 'prop-types';
import { Autocomplete, TextField, CircularProgress } from '@mui/material';
import { searchCatalog } from '../../api';

/**
 * Autocomplete input that dynamically fetches tag suggestions (BT, GK, Category, Region, Flavor)
 * based on the selected condition or action type and the active rule domain.
 * @param {object} props - Component properties.
 * @return {React.JSX.Element}
 */
export default function TagAutocomplete({ value, onChange, type, ruleDomain, placeholder }) {
  const [options, setOptions] = useState([]);
  const [inputValue, setInputValue] = useState(value || '');
  const [loading, setLoading] = useState(false);

  // Determine dataset and domain based on condition/action type and ruleDomain
  const { dataset, domains } = useMemo(() => {
    if (type === 'bt_is' || type === 'set_bt') {
      return {
        dataset: 'bt',
        domains: ruleDomain ? [ruleDomain] : [],
      };
    }
    if (type === 'gk_contains' || type === 'add_gk' || type === 'remove_gk') {
      return {
        dataset: 'gk',
        domains: ruleDomain ? [ruleDomain] : [],
      };
    }
    if (type === 'region_is' || type === 'set_region') {
      return {
        dataset: 'category', // Region tags are fetched via dataset='category' on food domain
        domains: ['food'],
      };
    }
    if (type === 'category_contains' || type === 'set_category') {
      return {
        dataset: 'category', // Category tags are fetched via dataset='category' on market domain
        domains: ['market'],
      };
    }
    if (type === 'flavor_contains') {
      return {
        dataset: 'brands',
        domains: ['food'],
      };
    }
    return { dataset: null, domains: [] };
  }, [type, ruleDomain]);

  useEffect(() => {
    if (!dataset || domains.length === 0) {
      setOptions([]);
      return;
    }

    let active = true;

    const fetchSuggestions = async () => {
      setLoading(true);
      try {
        const promises = domains.map((dom) =>
          searchCatalog({
            dataset,
            domain: dom,
            query: inputValue,
            page_size: 50,
          })
        );
        const responses = await Promise.all(promises);

        if (!active) return;

        const mergedResults = [];
        const seenNames = new Set();

        responses.forEach((res) => {
          if (res && res.results) {
            res.results.forEach((item) => {
              const nameLower = item.name.toLowerCase().trim();
              if (!seenNames.has(nameLower)) {
                seenNames.add(nameLower);
                mergedResults.push(item.name);
              }
            });
          }
        });

        mergedResults.sort((a, b) => a.localeCompare(b));
        setOptions(mergedResults);
      } catch (err) {
        console.error('Failed to fetch autocomplete suggestions:', err);
      } finally {
        if (active) setLoading(false);
      }
    };

    const timer = setTimeout(() => {
      fetchSuggestions();
    }, 400);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [inputValue, dataset, domains]);

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
