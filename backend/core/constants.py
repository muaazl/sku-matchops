"""Domain and catalog schema constants for the Backend API Gateway."""

DOMAIN_MARKET = "market"
DOMAIN_FOOD = "food"
DOMAINS = (DOMAIN_MARKET, DOMAIN_FOOD)

AUTO_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.50
CACHE_SALT = "MatchOps_v2"

FOOD_CATALOG_SHEET = "Food Catalog"
MARKET_CATALOG_SHEET = "Market Catalog"

CATALOG_COL_MAP_FOOD = {
    "Dish Name": "Name",
    "Item Name": "Name",
    "Description": "Description",
    "GenericKeywords": "Generic keywords",
    "Generic Keywords": "Generic keywords",
    "Generic keywords": "Generic keywords",
    "BasicType": "basictype",
    "Basic Type": "basictype",
    "Basic_Type": "basictype",
    "Flavor": "Flavor",
    "Flavour": "Flavor",
    "Price": "Price",
    "Region": "region",
    "Category": "region",
}

CATALOG_COL_MAP_MARKET = {
    "Item Name": "Name",
    "Name": "Name",
    "Description": "Description",
    "GenericKeywords": "Generic keywords",
    "Generic Keywords": "Generic keywords",
    "Generic keywords": "Generic keywords",
    "BasicType": "basictype",
    "Basic Type": "basictype",
    "Basic_Type": "basictype",
    "Brand": "Brand",
    "Category": "category",
    "Price": "Price",
}


def get_third_tag_col(domain: str = DOMAIN_MARKET) -> str:
    """Returns the third taxonomy column name for the given domain."""
    return "region" if domain == DOMAIN_FOOD else "category"
