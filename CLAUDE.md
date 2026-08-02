# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LogAI is a Python library for log analytics and intelligence (parsing, clustering, summarization, anomaly
detection — both statistical/ML and deep-learning). It follows the OpenTelemetry log data model and exposes
a Dash-based GUI on top of the core library.

## Setup and commands

```shell
# Install core + dev extras (editable install recommended while developing)
pip install -e ".[dev]"

# Optional extras
pip install -e ".[deep-learning]"   # torch/transformers-based algorithms (LSTM, CNN, Transformer, LogBERT, forecast_nn)
pip install -e ".[gui]"             # dash / plotly for the GUI portal
pip install -e ".[all]"             # everything

# NLTK punkt is required at runtime and not installed automatically
python -m nltk.downloader punkt
```

Run tests:
```shell
./run_unittests.sh                       # sets PYTHONPATH=./tests/ and runs pytest over the whole repo
python3 -m pytest tests/logai/algorithms/parsing_algo/test_drain.py                        # single file
python3 -m pytest tests/logai/algorithms/parsing_algo/test_drain.py::TestDrain::test_fit_parse  # single test
```
Tests rely on fixture data under `tests/logai/test_data/` (sample HDFS/BGL/HealthApp logs and pre-computed
intermediate results such as `default_parsed_result`, `default_feature_set`) and shared fixtures in
`tests/logai/test_utils/fixtures.py`. Tests that need torch/transformers will fail unless the
`deep-learning` extra is installed.

Lint/format (operates only on files changed vs. a base branch, default `origin/main`):
```shell
./run_black.sh [base-branch]
./run_flake8.sh [base-branch]
```
flake8 config: max line length 120, `E203` ignored (see `setup.cfg`), matching Black's style.

Run the GUI locally:
```shell
export PYTHONPATH='.'
python3 gui/application.py   # Dash server at http://localhost:8050
```

The GUI has a few environment gotchas not covered by `pip install -e ".[gui]"` alone:
- **Use Python 3.10, not 3.11+.** `logai/dataloader/data_model.py` declares a `@dataclass` field with a
  `pandas.DataFrame` default. Python's `dataclasses` module added a stricter mutable-default check in 3.11
  that rejects any unhashable default (not just `list`/`dict`/`set`), so importing the package at all raises
  `ValueError: mutable default <class 'pandas.DataFrame'> for field ... is not allowed` on 3.11/3.12/3.13.
- **Also install the `deep-learning` extra**, even though it's documented as optional:
  `logai/applications/application_interfaces.py` unconditionally imports
  `logai.analysis.nn_anomaly_detector`, which imports `datasets`/`torch` at module level, so `gui/` fails to
  import without them (`pip install -e ".[gui,deep-learning]"`).
- **Pin `dash<3`.** `setup.py` only requires `dash>=2.5.1`, so a fresh install pulls Dash 4.x, but
  `gui/application.py` calls `app.run_server(...)`, an API removed in Dash 3+. Use
  `pip install "dash>=2.5.1,<3"` after the editable install.

Build docs (Sphinx):
```shell
cd docs && make clean && make html
```

Update license headers (BSD-3-Clause, Salesforce):
```shell
python -m licenseheaders -t .copyright.tmpl -y "2023" -o "Salesforce.com, inc."
```

## Architecture

### Pipeline shape

LogAI applications compose the same set of stage modules under `logai/` into different pipelines depending
on the task. `LogClustering` and `LogAnomalyDetection` run the full chain:

```
DataLoader / OpenSetDataLoader  ->  Preprocessor  ->  LogParser  ->  LogVectorizer
    ->  CategoricalEncoder + FeatureExtractor  ->  Clustering / AnomalyDetector
```

`AutoLogSummarization` only needs the first three stages (DataLoader -> Preprocessor -> LogParser) since it
groups raw logs by parsed pattern rather than building feature vectors. `NNAnomalyDetector` (deep-learning
path) skips the classical feature-extraction stage entirely and trains/predicts directly on
`ForecastNNVectorizedDataset`/HuggingFace `Dataset` objects produced by the neural vectorization algorithms
in `logai/algorithms/vectorization_algo/forecast_nn.py` / `logbert.py`.

- `logai/dataloader/data_loader.py` (`FileDataLoader`) and `openset_data_loader.py` (`OpenSetDataLoader`)
  load raw logs (custom files or public benchmark datasets like HDFS/BGL/HealthApp/Thunderbird) into a
  `LogRecordObject`.
- `logai/dataloader/data_model.py` defines `LogRecordObject`, the central data structure modeled on the
  OpenTelemetry log/event record spec (`timestamp`, `attributes`, `resource`, `trace_id`, `span_id`,
  `severity_text/number`, `body`, `labels`, all indexed pandas DataFrames that must share a common index).
  It supports `to_dataframe()/from_dataframe()`, `save_to_csv()/load_from_csv()` (data + a `_metadata.json`
  sidecar describing which columns map to which field), and index-based `select_by_index`/`filter_by_index`.
- `logai/preprocess/` cleans raw loglines (`Preprocessor`) and partitions data (`Partitioner`,
  `OpenSetPartitioner`); dataset-specific preprocessors (`hdfs_preprocessor.py`, `bgl_preprocessor.py`,
  `thunderbird_preprocessor.py`) handle quirks of each public dataset.
- `logai/information_extraction/` turns cleaned loglines into structured features: `log_parser.py`
  (template/parameter extraction), `log_vectorizer.py` (embeds parsed loglines), `categorical_encoder.py`
  (encodes structured attributes), `feature_extractor.py` (assembles the final feature matrix, e.g. grouped
  by category/time window).
- `logai/analysis/` runs the actual task: `clustering.py`, `anomaly_detector.py` (classical ML on feature
  matrices), and `nn_anomaly_detector.py` (deep-learning models operating on
  `ForecastNNVectorizedDataset`/HuggingFace `Dataset` objects instead of feature matrices).
- `logai/applications/` are the top-level orchestrators end users instantiate (`LogClustering`,
  `LogAnomalyDetection`, `AutoLogSummarization`). Each takes a `WorkFlowConfig` (`application_interfaces.py`)
  and its `execute()` runs the full pipeline above, exposing results via properties. `applications/openset/`
  holds variants used for benchmarking anomaly detection on public datasets.

### Config pattern

Every stage has a paired `attr`/`dataclass` config extending `Config` in `logai/config_interfaces.py`, which
provides `from_dict()` (populates only fields present in the dict, ignores the rest) and `as_dict()`. Nested
configs are composed and reconstructed manually — see `WorkFlowConfig.from_dict()` in
`logai/applications/application_interfaces.py`, which is the top-level config aggregating every sub-config
(`data_loader_config`, `preprocessor_config`, `log_parser_config`, `log_vectorizer_config`,
`feature_extractor_config`, `categorical_encoder_config`, `anomaly_detection_config`,
`nn_anomaly_detection_config`, `clustering_config`, etc.). Workflows are typically defined as JSON/YAML and
loaded via `WorkFlowConfig.from_dict(json.loads(...))`.

### Algorithm interfaces + factory/registry

`logai/algorithms/algo_interfaces.py` defines the abstract base classes every algorithm implementation must
satisfy: `ParsingAlgo`, `VectorizationAlgo`, `FeatureExtractionAlgo`, `ClusteringAlgo`, `AnomalyDetectionAlgo`,
`NNAnomalyDetectionAlgo`, `CategoricalEncodingAlgo`. Concrete implementations live under
`logai/algorithms/{parsing,vectorization,clustering,categorical_encoding,anomaly_detection}_algo/` and
`nn_model/` (transformers, LSTM/CNN forecast models, LogBERT).

New algorithms register themselves with the singleton `logai.algorithms.factory.factory`
(`AlgorithmFactory`) via the `@factory.register(task, name, config_class)` class decorator, where `task` is
one of `detection`, `parsing`, `clustering`, `vectorization`. The factory maps an algorithm name string (as
used in JSON/YAML configs, e.g. `"parsing_algorithm": "drain"`) to its `(config_class, algo_class)` pair, and
`get_algorithm()`/`get_config()` are how the higher-level modules instantiate the right class dynamically.
Algorithms named in `_algorithms_with_torch` (`lstm`, `cnn`, `transformer`, `logbert`, `forecast_nn`) raise an
`ImportError` pointing at `logai[deep-learning]` if torch/transformers aren't installed — check
`logai/utils/misc.py` (`is_torch_available`, `is_transformers_available`, `is_tf_available`,
`is_nltk_available`) before assuming an optional dependency is present.

### GUI

`gui/` is a separate Plotly Dash application layered on top of the core library (`application.py` entry
point, `callbacks/`, `pages/`, `file_manager.py`). It requires `PYTHONPATH=.` and the `gui` extra, and is
tested independently under `tests/gui/`.
