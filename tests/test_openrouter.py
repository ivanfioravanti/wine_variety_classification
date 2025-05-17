import importlib
from unittest.mock import patch, Mock
import pandas as pd
from config import SAMPLE_SIZE


def dummy_df(n=SAMPLE_SIZE):
    return pd.DataFrame({
        "country": ["France"] * n,
        "variety": ["Chardonnay"] * n,
        "winery": ["w"] * n,
        "province": ["p"] * n,
        "region_1": ["r"] * n,
        "description": ["d"] * n,
        "taster_name": ["t"] * n,
        "points": [90] * n,
        "price": [20] * n,
    })


def test_call_model_returns_chardonnay():
    df = dummy_df()
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "{\"variety\": \"Chardonnay\"}"}}]
    }
    with patch("pandas.read_csv", return_value=df):
        with patch("requests.post", return_value=mock_response) as mock_post:
            # Reload module with patched pandas.read_csv
            import sys
            if "providers.wine_openrouter" in sys.modules:
                del sys.modules["providers.wine_openrouter"]
            openrouter = importlib.import_module("providers.wine_openrouter")
            result = openrouter.call_model("model", "prompt")
            assert result == "Chardonnay"
            mock_post.assert_called_once()
