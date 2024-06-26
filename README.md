# zooms_search

Script to batch search Bruker peaklist.xml files of PMF of collagens against [ZooMS marker data](https://docs.google.com/spreadsheets/d/1ipm9fFFyha8IEzRO2F5zVXIk0ldwYiWgX5pGqETzBco/edit#gid=1005946405)

## Requirements
### Python 3.*
(tested and developed with 3.12.3)
* pandas
* openpyxl
* numpy
* matplotlib

## How to use

Input data can be provided in form of a folder with multiple sub-folders containing the raw & processed data as generated by Bruker flexControl & flexAnalysis. Script scans for peaklist.xml files and extract sample names from proc file, procs file, or folder names. Output will contain:

* One excel file containing an "overview" of all samples with the top hit(s) found for each sample, in case of same number of matched peaks for the two best hits, the script adds both into this excel file
* For each sample a folder with more detailed results:
  * An excel file with all hits found, sorted by number of matched peaks
  * An excel file containing the peaks extracted from the peaklist.xml file (for manual checking/matching)
  * A plot of the peaks in peaklist.xml with highlighted machted peaks of the "top-hit"

Run the script search_zooms.py from a CLI with python:

```
usage: search_zooms.py [-h] [--min_matched_peaks MIN_MATCHED_PEAKS] [--max_match_error MAX_MATCH_ERROR] [--peak_filter PEAK_FILTER [PEAK_FILTER ...]] [--bruker_peaklist_filename BRUKER_PEAKLIST_FILENAME] input output zooms_db

search_zooms - Script to batch search Bruker peaklist.xml files agains ZooMS marker data

positional arguments:
  input                 folder with input files, expecting Bruker flexControl format, peaklist.xml
  output                folder for output files
  zooms_db              path to the ZooMS Marker excel file

options:
  -h, --help            show this help message and exit
  --min_matched_peaks MIN_MATCHED_PEAKS
                        Minimum number of matched peaks to accept as potential hit - default 3
  --max_match_error MAX_MATCH_ERROR
                        Max error in Da to accept a peak match - default 1.2
  --peak_filter PEAK_FILTER [PEAK_FILTER ...]
                        Add one or multiple filters (separated by whitespace) to filter peaklist.xml before searching for hits. Example: --filter goodn2>=0.7 s2n>=5
  --bruker_peaklist_filename BRUKER_PEAKLIST_FILENAME
                        Name of the Bruker peak list files - default peaklist.xml
```
