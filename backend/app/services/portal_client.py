import io
from typing import Dict, List, Union

import pandas as pd
import polars as pl
import requests


class PortalClient:
    """Fetches merchant SKU CSVs from the PickMe Food Portal API."""

    BASE_URL = "https://food-portal-api-go.pickme.lk/v1/food/place/skus/csv"

    def __init__(self, bearer_token: str):
        if not bearer_token or not bearer_token.strip():
            raise ValueError("Bearer token is required.")
        self.headers = {"Authorization": f"Bearer {bearer_token.strip()}"}

    def fetch_merchant_csv(self, merchant_id: str) -> pd.DataFrame:
        """
        Fetches the SKU CSV for a single merchant ID.
        Returns a DataFrame or raises on failure.
        """
        url = f"{self.BASE_URL}/{merchant_id.strip()}"
        response = requests.get(url, headers=self.headers, timeout=30)

        if response.status_code == 401:
            raise PermissionError("Authentication failed — token may be expired or invalid.")
        if response.status_code == 404:
            raise FileNotFoundError(f"Merchant ID '{merchant_id}' not found on portal.")

        response.raise_for_status()

        # Parse CSV from response body using Polars for memory-efficiency
        try:
            df_pl = pl.read_csv(response.content, ignore_errors=True)
            if df_pl.is_empty():
                raise ValueError(f"Merchant '{merchant_id}' returned an empty CSV.")
            return df_pl.to_pandas()
        except Exception as e:
            if "returned an empty CSV" in str(e):
                raise e
            df = pd.read_csv(io.StringIO(response.text), encoding_errors="replace", on_bad_lines="skip")
            if df.empty:
                raise ValueError(f"Merchant '{merchant_id}' returned an empty CSV.")
            return df

    def fetch_multiple(self, merchant_ids: List[str]) -> Dict[str, Union[pd.DataFrame, str]]:
        """
        Fetches CSVs for multiple merchant IDs.
        Returns a dict of {merchant_id: DataFrame | error_string}.
        """
        results = {}
        for mid in merchant_ids:
            mid = mid.strip()
            if not mid:
                continue
            try:
                results[mid] = self.fetch_merchant_csv(mid)
            except Exception as e:
                results[mid] = f"Error: {e}"
        return results
