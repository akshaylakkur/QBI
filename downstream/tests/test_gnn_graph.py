"""Tests for GNN graph construction."""

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from downstream.gnn.graph import build_graph_from_dataframe, graphs_from_exposure_csv


def _mini_df():
    return pd.DataFrame(
        [
            {
                "tomo_id": "T1",
                "particle_type": "beta-galactosidase",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "radius_angstrom": 90.0,
                "steric_exposure": 1.0,
            },
            {
                "tomo_id": "T1",
                "particle_type": "ribosome",
                "x": 300.0,
                "y": 0.0,
                "z": 0.0,
                "radius_angstrom": 150.0,
                "steric_exposure": 0.7,
            },
        ]
    )


def test_build_graph_shapes():
    g = build_graph_from_dataframe(_mini_df(), "T1")
    assert g.num_nodes == 2
    assert g.x.shape[0] == 2
    assert g.y.shape == (2, 1)
    assert g.edge_index.shape[0] == 2
    assert g.edge_attr.shape[1] == 4


def test_graphs_from_csv(tmp_path):
    path = tmp_path / "labels.csv"
    _mini_df().to_csv(path, index=False)
    graphs = graphs_from_exposure_csv(path)
    assert len(graphs) == 1
    assert graphs[0].tomo_id == "T1"
