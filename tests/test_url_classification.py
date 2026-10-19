import unittest

import main as script
from url_classification import (
    NEW_URLS,
    TRAINING_DATA,
    classificar_urls,
    limpar_url,
    treinar_modelo,
)


class UrlClassificationTests(unittest.TestCase):
    def test_training_data_fits_and_classifies_each_url(self) -> None:
        modelo, vetorizador = treinar_modelo()

        self.assertEqual(len(TRAINING_DATA), 12)
        self.assertEqual(
            classificar_urls(NEW_URLS, modelo, vetorizador),
            ["Sports", "News", "Finance", "Other"],
        )

    def test_cleaning_preserves_digits(self) -> None:
        self.assertEqual(limpar_url("https://www.site123.com/path42"), "site123 com path42")

    def test_main_script_classifies_demo_urls(self) -> None:
        modelo, vetorizador = script.treinar_modelo()

        self.assertEqual(
            script.classificar_urls(script.NEW_URLS, modelo, vetorizador),
            ["News", "Sports", "News", "Finance", "Other"],
        )

    def test_classifying_no_urls_returns_empty_list(self) -> None:
        modelo, vetorizador = treinar_modelo()

        self.assertEqual(classificar_urls([], modelo, vetorizador), [])


if __name__ == "__main__":
    unittest.main()
