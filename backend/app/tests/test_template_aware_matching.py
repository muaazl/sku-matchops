import unittest
from engine import config

class TestTemplateAwareMatching(unittest.TestCase):
    def test_flag_defaults(self):
        self.assertTrue(getattr(config, "ENABLE_TEMPLATE_TAG_ENRICHMENT", False))
        self.assertFalse(getattr(config, "ALLOW_UNREGISTERED_TEMPLATE_KEYWORDS", True))
        self.assertTrue(getattr(config, "ENABLE_BRAND_STRIPPED_SEARCH", False))

if __name__ == "__main__":
    unittest.main()
