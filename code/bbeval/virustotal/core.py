import vt


class VisusTotal:
    def __init__(self):
        """Constants for the VirusTotal API."""
        # Make sure we delete all of these before making the API public
        self._API_KEY = "3a062f56cbf28b0400100068fecff1145f95b74bdbf15a0993ca7cd70b8a3f24"
        self._LIMIT_PER_MINUTE = 4
        self._NUM_ENGINES = 76
        self._client = vt.Client(self._API_KEY)
    
    def get_preds(self, x):
        analysis = self._client.scan_file(x, wait_for_completion=True)
        results = {}
        for k, v in analysis.stats.items():
            results[k] = v['category']
        assert len(results) == self._NUM_ENGINES, "Not all engines returned a result"
        return results
