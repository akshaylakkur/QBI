def __getattr__(name):
    import importlib

    if name in ("build_graph_from_dataframe", "graphs_from_exposure_csv", "fit_feature_stats_on_graphs"):
        return getattr(importlib.import_module(".graph", __package__), name)
    if name == "ExposureGNN":
        return getattr(importlib.import_module(".model", __package__), name)
    if name in ("run_pipeline", "TrainConfig"):
        return getattr(importlib.import_module(".train", __package__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "build_graph_from_dataframe",
    "graphs_from_exposure_csv",
    "ExposureGNN",
    "run_pipeline",
    "TrainConfig",
]
