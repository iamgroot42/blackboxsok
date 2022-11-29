import vt
import numpy as np
import time
from tqdm import tqdm


class VisusTotal:
    def __init__(self):
        """Constants for the VirusTotal API."""
        # Make sure we delete all of these before making the API public
        self._API_KEY = "3a062f56cbf28b0400100068fecff1145f95b74bdbf15a0993ca7cd70b8a3f24"
        self._client = vt.Client(self._API_KEY)
    
    def get_preds(self, paths):
        batch_analyses = []
        results = np.zeros((len(paths), 2))
        completed = np.zeros(len(paths), dtype=bool)
        # Send all requests as a batch
        for path in tqdm(paths, desc="Sending requests"):
            with open(path, "rb") as f:
                analysis = self._client.scan_file(f, wait_for_completion=False)
                batch_analyses.append(analysis)
        batch_analyses = np.array(batch_analyses, dtype=object)
        # Sleep for 30 seconds
        time.sleep(30)
        # Iterate over results
        with tqdm(total=len(paths), desc="Collecting responses") as pbar:
            while not completed.all():
                not_completed = np.where(~completed)[0]
                analyses = [self._client.get_object("/analyses/{}", ba.id) for ba in batch_analyses[not_completed]]
                for i, a in enumerate(analyses):
                    if a.status == "completed":
                        completed[not_completed[i]] = True
                        pbar.update(1)
                        results[not_completed[i]] = self._get_nums(a.stats)
                # Break if we're done
                if completed.all():
                    break
                # Sleep for 45s
                time.sleep(45)
        # Return results
        return results

    def _get_nums(self, stats):
        malicious = stats['malicious'] + stats['suspicious']
        undetected = stats['undetected'] + stats['harmless']
        return np.array([undetected, malicious])
