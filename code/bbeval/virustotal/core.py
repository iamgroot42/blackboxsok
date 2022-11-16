import vt
import numpy as np
from tqdm import tqdm


class VisusTotal:
    def __init__(self):
        """Constants for the VirusTotal API."""
        # Make sure we delete all of these before making the API public
        self._API_KEY = "3a062f56cbf28b0400100068fecff1145f95b74bdbf15a0993ca7cd70b8a3f24"
        self._LIMIT_PER_MINUTE = 4
        self._client = vt.Client(self._API_KEY)
    
    def _get_pred(self, path):
        with open(path, "rb") as f:
            analysis = self._client.scan_file(f, wait_for_completion=True)
        malicious = analysis.stats['malicious'] + analysis.stats['suspicious']
        undetected = analysis.stats['undetected'] + analysis.stats['harmless']
        return np.array([undetected, malicious])

    def get_preds(self, paths):
        all_preds = []
        for path in tqdm(paths, desc="Getting predictions from VirusTotal"):
            all_preds.append(self._get_pred(path))
        return np.array(all_preds)
