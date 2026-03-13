import requests
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    alljobs = []
    url_base = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/fileServer/birdhouse/testdata/ORRC/netcdf/{subfolder}/{date}.nc"
    dates = pd.date_range(start="2026-01-01", end="2026-01-09", freq='6h').strftime("%Y%m%d%H").tolist()
    for subfolder in ["RDPS", "CaPA_coarse"]:
        for date in dates:
            request = url_base.format(subfolder=subfolder, date=date)
            output_file = f"code/netcdf/{subfolder}/{date}.nc"
            #download_one(output_file, subfolder, request)
            alljobs.append((output_file, subfolder, request))
    
    if alljobs:
        for ntry in range(5):  # try up to 5 times
            # download files in parallel
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(download_one, *job) for job in alljobs]
                for future in as_completed(futures):
                    result = future.result()
                # check if any jobs failed (result is None)
                failed_jobs = [job for job, result in zip(alljobs, futures) if result is None]
                if not failed_jobs:
                    print("All downloads succeeded.")
                    break  # all jobs succeeded, exit retry loop
                else:
                    alljobs = failed_jobs  # retry only the failed jobs
                if failed_jobs:
                    print(f"Following downloads failed after 5 attempts: {[job[0] for job in failed_jobs]}")
        


def download_one(output_file, dataset, request):
    """ Download a single file."""
    output_file = Path(output_file)
    if output_file.exists():
        print(f"Skipping existing file: {output_file}")
        return output_file
    try:
        
        print(f"Downloading {output_file}")
        response = requests.get(request, stream=True, timeout=30)  # Set a timeout for the request
        response.raise_for_status()  # Check if the request was successful
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
        with open(output_file.with_suffix('.tmp'), 'wb') as f:
            f.write(response.content)
        output_file.with_suffix('.tmp').rename(output_file)
        print(f"Downloaded {output_file}")
        return output_file
    except Exception as e:
        print(f"Failed to download {output_file}: {e}")
        return None
    
if __name__ == "__main__":
    main()
