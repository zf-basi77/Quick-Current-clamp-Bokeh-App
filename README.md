# Quick-Current-clamp-Bokeh-App


A Python application for interactive annotation and peak detection in time-series data. Built with Bokeh, NumPy, pandas, and SciPy, the tool provides a graphical interface for reviewing signals, recording points of interest, and saving annotations for downstream analysis.

## Features

- **Interactive visualization:** Load and inspect time-series data through a browser-based interface.
- **Point annotation:** Record coordinates for predefined features, including action potential peaks, thresholds, and afterhyperpolarization (AHP).
- **Peak detection:** Identify peaks using adjustable height, prominence, and minimum-distance parameters.
- **Annotation management:** Review recorded annotations and export results to CSV.
- **Persistent results:** Save annotations across files for subsequent analysis.

## Input Data

Input files should be CSV files containing two columns:

| Column | Description |
|---|---|
| `x` | Independent variable (e.g., time) |
| `y` | Signal amplitude |

## Requirements

- Python 3.9+
- NumPy
- pandas
- SciPy
- Bokeh

Install the dependencies with:

```bash
pip install numpy pandas scipy bokeh
```

## Usage

1. Set `DATA_DIR` in the script to the directory containing the input CSV files.
2. Run the application using the Bokeh server:

   ```bash
   bokeh serve --show annotator.py
   ```

3. Select a file and annotation type from the interface.
4. Click the signal to record point annotations, or select **Peak Count** to adjust peak detection parameters.
5. Save the results to export the annotation table to CSV.

## Output

The application generates `click_results.csv`, containing the input filename and associated annotations. Peak counts and their corresponding coordinates are stored alongside point annotations.

## Technologies

Python · Bokeh · NumPy · pandas · SciPy
