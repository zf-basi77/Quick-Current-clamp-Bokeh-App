
import os
import json
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.signal import find_peaks

from bokeh.io import curdoc
from bokeh.models import (
    ColumnDataSource, Select, Button, Div, Slider
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.events import Tap
from bokeh.palettes import Category10


# ============================================================
# CONFIGURATION
# ============================================================

# Directory containing the input traces and annotation results.
DATA_DIR = r'M:\Patch\bokh_test'
RESULTS_FILE = os.path.join(DATA_DIR, 'click_results.csv')

# Defines the available annotation types and their storage requirements.
# Point annotations record selected coordinates; peak detection identifies
# multiple local maxima within a trace.
METRICS = OrderedDict([
    ('AHP',        {'mode': 'point', 'fields': ['x']}),
    ('Threshold',  {'mode': 'point', 'fields': ['x', 'y']}),
    ('Peak',       {'mode': 'point', 'fields': ['x', 'y']}),
    ('Peak Count', {'mode': 'peaks'}),
])

# Assign consistent colors to point annotation types.
METRIC_COLORS = dict(zip(METRICS.keys(), Category10[10]))


# ============================================================
# RESULTS STORAGE
# ============================================================

# Define a consistent schema for the annotation output.
# AP waveforms and their associated measurements are serialized as JSON
# strings so that nested arrays can be stored in individual CSV cells.
RESULT_COLUMNS = ['file']

for name, cfg in METRICS.items():
    if cfg['mode'] == 'point':
        for field in cfg['fields']:
            RESULT_COLUMNS.append(f"{name}_{field}")

    elif cfg['mode'] == 'peaks':
        RESULT_COLUMNS += [
            "Peak Count_count",
            "Peak Count_xs",
            "Peak Count_ys",
            "AP_Count",
            "AP_PeakTimes",
            "AP_PeakVoltages",
            "AP_AccelerationTimes",
            "AP_AccelerationVoltages",
            "AP_TraceTimes",
            "AP_TraceVoltages"
        ]


def load_results():
    """Load existing annotations and accommodate changes to the output schema."""
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)

        # Preserve compatibility with results files created by earlier versions.
        for column_name in RESULT_COLUMNS:
            if column_name not in df.columns:
                df[column_name] = np.nan

        return df[RESULT_COLUMNS]

    return pd.DataFrame(columns=RESULT_COLUMNS)


# Main results source shared by the application callbacks and interface.
results_source = ColumnDataSource(load_results().to_dict('list'))


# ============================================================
# LIVE DATA SOURCES
# ============================================================

# Currently selected trace and its manually recorded annotations.
plot_source = ColumnDataSource(data=dict(x=[], y=[]))
marker_source = ColumnDataSource(
    data=dict(x=[], y=[], metric=[], color=[])
)

# Peak detections displayed while adjusting the detection parameters.
peak_source = ColumnDataSource(data=dict(x=[], y=[]))

# Data displayed in the firing-rate versus trace-number plot.
fi_source = ColumnDataSource(
    data=dict(trace=[], frequency=[], file=[])
)


# ============================================================
# DATA AND INTERFACE HELPERS
# ============================================================

def show_peak_controls():
    """Display the controls used to adjust peak detection parameters."""
    peak_controls.children = [
        height_slider,
        prom_slider,
        dist_slider,
        commit_btn
    ]


def set_value(file, column, value):
    """Update a single field in the results row associated with a file."""
    data = results_source.data

    if file not in data['file']:
        return

    idx = data['file'].index(file)
    data[column][idx] = value

    # Notify Bokeh that the underlying results data have changed.
    results_source.trigger('data', data, data)


def get_or_create_row(file):
    """Ensure that the results table contains an entry for the selected file."""
    data = results_source.data

    if file in data['file']:
        return

    # Initialize unrecorded annotations as missing values.
    new_row = {column_name: np.nan for column_name in RESULT_COLUMNS}
    new_row['file'] = file

    for column_name in data:
        data[column_name].append(new_row.get(column_name, np.nan))

    results_source.trigger('data', data, data)


def nearest_point(x, y):
    """Return the sampled coordinate closest to a selected plot location."""
    xs = np.asarray(plot_source.data['x'])
    ys = np.asarray(plot_source.data['y'])

    if len(xs) == 0:
        return None

    # Snap clicks to recorded samples rather than interpolating the signal.
    idx = np.argmin((xs - x)**2 + (ys - y)**2)

    return float(xs[idx]), float(ys[idx])


def y_from_x(xv):
    """Retrieve the signal amplitude at the sample nearest to xv."""
    xs = np.asarray(plot_source.data['x'])
    ys = np.asarray(plot_source.data['y'])

    if len(xs) == 0:
        return np.nan

    idx = np.argmin(np.abs(xs - xv))
    return float(ys[idx])


# ============================================================
# ANNOTATION DISPLAY
# ============================================================

def refresh_markers():
    """Refresh the plot markers using annotations saved for the active file."""
    data = results_source.data

    if file_select.value not in data['file']:
        marker_source.data = dict(x=[], y=[], metric=[], color=[])
        return

    idx = data['file'].index(file_select.value)

    xs, ys, metrics, colors = [], [], [], []

    for metric, cfg in METRICS.items():
        # Peak detections are displayed separately from point annotations.
        if cfg['mode'] != 'point':
            continue

        x_column = f"{metric}_x"
        y_column = f"{metric}_y"

        if x_column in data and not pd.isna(data[x_column][idx]):
            x_value = data[x_column][idx]
            xs.append(x_value)

            # For annotations requiring only an x-coordinate, recover the
            # corresponding y-value from the original sampled trace.
            if y_column in data and not pd.isna(data[y_column][idx]):
                ys.append(data[y_column][idx])
            else:
                ys.append(y_from_x(x_value))

            metrics.append(metric)
            colors.append(METRIC_COLORS[metric])

    marker_source.data = dict(
        x=xs,
        y=ys,
        metric=metrics,
        color=colors
    )


# ============================================================
# POINT ANNOTATION
# ============================================================

def record_point(metric, x, y):
    """Record a selected coordinate and update the annotation display."""
    file = file_select.value
    get_or_create_row(file)

    cfg = METRICS[metric]

    # Store only the coordinates required by the selected annotation type.
    if 'x' in cfg['fields']:
        set_value(file, f"{metric}_x", x)

    if 'y' in cfg['fields']:
        set_value(file, f"{metric}_y", y)

    refresh_markers()
    update_active_view()

    status_div.text = f"Recorded {metric}: x={x:.3f}"


# ============================================================
# PEAK DETECTION
# ============================================================

def compute_peaks():
    """Detect peaks within the central region of the selected trace."""
    x = np.asarray(plot_source.data['x'])
    y = np.asarray(plot_source.data['y'])

    # Restrict detection to a fixed-width window around the trace midpoint.
    # The window is expressed in samples, not in units of the x-axis.
    center = len(y) // 2
    window = 20000

    start = max(0, center - window)
    end = min(len(y), center + window)

    # Apply the current amplitude, prominence, and minimum-distance criteria.
    local_indices = find_peaks(
        y[start:end],
        height=height_slider.value,
        prominence=prom_slider.value,
        distance=int(dist_slider.value)
    )[0]

    # Convert indices from the restricted window to the original trace.
    indices = local_indices + start

    return x[indices], y[indices]


def update_peaks():
    """Refresh the displayed peak markers using the current slider settings."""
    peak_x, peak_y = compute_peaks()

    peak_source.data = dict(
        x=list(peak_x),
        y=list(peak_y)
    )


def commit_peaks():
    """Save the current peak count and detected peak coordinates."""
    file = file_select.value
    get_or_create_row(file)

    # Recalculate peaks before saving to ensure the results match the
    # current detection parameters.
    peak_x, peak_y = compute_peaks()

    peak_source.data = dict(
        x=list(peak_x),
        y=list(peak_y)
    )

    set_value(file, "Peak Count_count", len(peak_x))
    set_value(file, "Peak Count_xs", ";".join(map(str, peak_x)))
    set_value(file, "Peak Count_ys", ";".join(map(str, peak_y)))

    update_active_view()
    status_div.text = f"Committed {len(peak_x)} peaks."


# ============================================================
# INDIVIDUAL ACTION POTENTIAL ANALYSIS
# ============================================================

def analyze_committed_aps():
    """Extract individual AP waveforms and measure their maximum dV/dt points.

    Each detected peak is used as the center of an extraction window extending
    up to 400 samples on either side. Complete windows contain 800 samples,
    with the peak at index 400. Windows near trace boundaries are truncated.
    """
    file = file_select.value

    if not file:
        return

    x = np.asarray(plot_source.data['x'], dtype=float)
    y = np.asarray(plot_source.data['y'], dtype=float)

    if len(x) < 2:
        status_div.text = "Not enough data points to analyze APs."
        return

    # Use the same detection algorithm and parameters as the interactive plot.
    peak_x, peak_y = compute_peaks()

    if len(peak_x) == 0:
        status_div.text = (
            "No peaks found. Adjust the peak settings and commit peaks first."
        )
        return

    # Process detected peaks in chronological order.
    order = np.argsort(peak_x)
    peak_x = np.asarray(peak_x)[order]
    peak_y = np.asarray(peak_y)[order]

    # Number of samples included on either side of each detected peak.
    window = 400

    ap_times = []
    ap_voltages = []
    acceleration_times = []
    acceleration_voltages = []

    for peak_time in peak_x:

        # Map the detected peak coordinate back to its sample index.
        peak_idx = int(np.argmin(np.abs(x - peak_time)))

        # Extract the local waveform, truncating windows at trace boundaries.
        left = max(0, peak_idx - window)
        right = min(len(x), peak_idx + window)

        trace_times = x[left:right]
        trace_voltages = y[left:right]

        # Retain the complete extracted waveform for downstream analysis.
        ap_times.append(trace_times.tolist())
        ap_voltages.append(trace_voltages.tolist())

        # Calculate the voltage derivative with respect to the x-axis.
        # The x-values must be strictly increasing for this calculation.
        if (
            len(trace_times) >= 2
            and np.all(np.diff(trace_times) > 0)
        ):
            dydt = np.gradient(trace_voltages, trace_times)

            # Identify the sample with the maximum positive voltage slope.
            max_slope_idx = int(np.argmax(dydt))

            # Store the time and voltage at maximum dV/dt.
            acceleration_times.append(
                float(trace_times[max_slope_idx])
            )
            acceleration_voltages.append(
                float(trace_voltages[max_slope_idx])
            )

        else:
            acceleration_times.append(np.nan)
            acceleration_voltages.append(np.nan)

    get_or_create_row(file)

    # Serialize arrays as JSON to preserve their nested structure in the CSV.
    set_value(file, "AP_Count", len(peak_x))
    set_value(file, "AP_PeakTimes", json.dumps(peak_x.tolist()))
    set_value(file, "AP_PeakVoltages", json.dumps(peak_y.tolist()))
    set_value(
        file,
        "AP_AccelerationTimes",
        json.dumps(acceleration_times)
    )
    set_value(
        file,
        "AP_AccelerationVoltages",
        json.dumps(acceleration_voltages)
    )
    set_value(file, "AP_TraceTimes", json.dumps(ap_times))
    set_value(file, "AP_TraceVoltages", json.dumps(ap_voltages))

    update_active_view()

    status_div.text = (
        f"Analyzed {len(peak_x)} APs and saved individual traces, "
        "peak voltages, and max-dV/dt times/voltages."
    )


# ============================================================
# FREQUENCY-INTENSITY (F-I) PLOT
# ============================================================

def generate_fi_plot():
    """Plot estimated firing frequency against trace number.

    Peak counts are converted to frequency using a fixed 0.5-second
    analysis duration. Trace numbers are extracted from filenames when
    possible and otherwise assigned according to their results-table index.
    """
    data = results_source.data
    rows = []

    for idx, filename in enumerate(data.get('file', [])):
        count = data["Peak Count_count"][idx]

        # Only include traces with a previously committed peak count.
        if pd.isna(count):
            continue

        # Extract the numeric suffix from filenames such as wave0.csv.
        match = re.search(r'(\d+)(?=\.csv$|$)', str(filename))
        trace_number = int(match.group(1)) if match else idx

        rows.append((trace_number, str(filename), float(count)))

    if not rows:
        status_div.text = (
            "No committed peak counts available. "
            "Commit peaks on the traces first."
        )
        return

    # Order traces numerically before constructing the F-I curve.
    rows.sort(key=lambda item: item[0])

    trace_numbers = [item[0] for item in rows]

    # Convert counts to Hz using the assumed 0.5-second recording duration.
    frequencies = [item[2] / 0.5 for item in rows]

    fi_source.data = dict(
        trace=trace_numbers,
        frequency=frequencies,
        file=[item[1] for item in rows]
    )

    # Create a separate figure for the firing-frequency summary.
    fi_plot = figure(
        width=850,
        height=450,
        title="F-I Plot",
        x_axis_label="Trace number",
        y_axis_label="Frequency (Hz)",
        tools="pan,wheel_zoom,reset,save"
    )

    fi_plot.line('trace', 'frequency', source=fi_source)
    fi_plot.scatter('trace', 'frequency', source=fi_source, size=9)

    fi_plot_holder.children = [fi_plot]
    status_div.text = f"Generated F-I plot from {len(rows)} traces."


# ============================================================
# ACTIVE FILE SUMMARY
# ============================================================

def update_active_view():
    """Display all saved annotation fields for the selected file."""
    data = results_source.data

    if file_select.value not in data['file']:
        active_row_div.text = "No file selected"
        return

    idx = data['file'].index(file_select.value)

    # Summarize point annotations, peak detections, and AP measurements.
    text = f"<b>{file_select.value}</b><br><br>"

    for column_name in data:
        if column_name != "file":
            text += f"{column_name}: {data[column_name][idx]}<br>"

    active_row_div.text = text


# ============================================================
# INTERFACE CALLBACKS
# ============================================================

def on_tap(event):
    """Handle plot clicks and record the nearest sampled coordinate."""
    if file_select.value is None:
        return

    point = nearest_point(event.x, event.y)

    if point is None:
        return

    record_point(metric_select.value, point[0], point[1])


def update_file(attr, old, new):
    """Load a new trace and initialize its display and detection parameters."""
    if not file_select.value:
        return

    df = pd.read_csv(os.path.join(DATA_DIR, file_select.value))

    # Update the source used by the main signal plot.
    plot_source.data = dict(
        x=df['x'].tolist(),
        y=df['y'].tolist()
    )

    # Scale the peak detection controls to the amplitude range of the trace.
    # A nonzero range is required for valid slider bounds and step sizes.
    y = np.asarray(df['y'])

    if len(y) == 0:
        ymin, ymax = 0, 1
    else:
        ymin, ymax = float(np.min(y)), float(np.max(y))

    if ymax == ymin:
        ymax = ymin + 1

    height_slider.start = ymin
    height_slider.end = ymax
    height_slider.value = ymin + 0.3 * (ymax - ymin)

    height_slider.step = (
        (ymax - ymin) / 200 if ymax > ymin else 0.1
    )
    prom_slider.step = (
        (ymax - ymin) / 200 if ymax > ymin else 0.1
    )
    dist_slider.step = 1

    prom_slider.start = 0
    prom_slider.end = ymax - ymin
    prom_slider.value = 0.1 * (ymax - ymin)

    # Restore saved annotations and update the active-file summary.
    refresh_markers()
    update_active_view()

    # Peak controls and markers are relevant only for peak-count analysis.
    if metric_select.value == "Peak Count":
        show_peak_controls()
        update_peaks()
    else:
        peak_controls.children = []


def on_metric_change(attr, old, new):
    """Update the interface when the selected annotation type changes."""
    if metric_select.value == "Peak Count":
        show_peak_controls()
        update_peaks()
    else:
        peak_controls.children = []


def on_peak_change(attr, old, new):
    """Recalculate the displayed peaks when a detection parameter changes."""
    update_peaks()


def save_results():
    """Write the current annotation table to the results CSV."""
    pd.DataFrame(results_source.data).to_csv(
        RESULTS_FILE,
        index=False
    )

    status_div.text = "Saved successfully."


# ============================================================
# USER INTERFACE
# ============================================================

# Populate the file selector with the available CSV files.
files = sorted([
    filename
    for filename in os.listdir(DATA_DIR)
    if filename.endswith(".csv")
])

file_select = Select(
    title="File",
    options=files,
    value=files[0] if files else None
)
file_select.on_change('value', update_file)

metric_select = Select(
    title="Metric",
    options=list(METRICS.keys()),
    value="AHP"
)
metric_select.on_change("value", on_metric_change)

# Status messages and the active-file annotation summary.
status_div = Div()
active_row_div = Div(width=800)


# Peak detection parameters are initialized when a trace is loaded.
height_slider = Slider(
    title="Height", start=0, end=1, value=0
)
prom_slider = Slider(
    title="Prominence", start=0, end=1, value=0
)
dist_slider = Slider(
    title="Distance", start=1, end=50, value=10
)

height_slider.on_change('value', on_peak_change)
prom_slider.on_change('value', on_peak_change)
dist_slider.on_change('value', on_peak_change)

commit_btn = Button(
    label="Commit Peaks",
    button_type="success"
)
commit_btn.on_click(commit_peaks)

save_btn = Button(
    label="Save",
    button_type="success"
)
save_btn.on_click(save_results)

# Analyze individual waveforms around detected peaks.
analyze_aps_btn = Button(
    label="Analyze Individual APs",
    button_type="success"
)
analyze_aps_btn.on_click(analyze_committed_aps)

# Generate the firing-frequency summary across recorded traces.
fi_btn = Button(
    label="Generate F-I Plot",
    button_type="primary"
)
fi_btn.on_click(generate_fi_plot)

# Container for the dynamically generated F-I figure.
fi_plot_holder = column()

# Holds the peak controls when peak-count analysis is selected.
peak_controls = column()


# ============================================================
# MAIN SIGNAL PLOT
# ============================================================

plot = figure(
    width=850,
    height=450,
    tools="pan,wheel_zoom,reset",
    output_backend="webgl"
)

# Original signal, manually selected points, and detected peaks are
# separate renderers so their data can be updated independently.
plot.line('x', 'y', source=plot_source)

plot.scatter(
    'x', 'y',
    source=marker_source,
    size=10,
    color='color'
)

plot.scatter(
    'x', 'y',
    source=peak_source,
    size=10,
    color="red",
    marker="triangle"
)

# Register the mouse-click callback for point annotation.
plot.on_event(Tap, on_tap)


# ============================================================
# APPLICATION LAYOUT
# ============================================================

layout = column(
    row(file_select, metric_select, save_btn),
    plot,
    peak_controls,
    row(commit_btn, analyze_aps_btn),
    fi_btn,
    fi_plot_holder,
    status_div,
    active_row_div
)

# Register the application layout with the Bokeh server.
curdoc().add_root(layout)
curdoc().title = "Annotator"
