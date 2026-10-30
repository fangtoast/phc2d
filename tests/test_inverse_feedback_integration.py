"""Integration of Fourier proposal, numerical evaluator and feedback state."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from phc2d.data.full_cell_spec import FullCellStructureSpec
from phc2d.generators.full_cell_latent import generate_unified_fourier_binary
from phc2d.inverse.feedback_evaluator import evaluate_feedback_candidate
from phc2d.inverse.feedback_loop import run_feedback_loop
from phc2d.inverse.feedback_state import start_feedback
from phc2d.inverse.fourier_feedback_proposal import propose_fourier_candidate
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


class ToyPredictor:
    def predict_tensor(self, raster):
        fill = raster.mean(dim=(1, 2))
        return torch.stack((.2+.1*fill, .8-.1*fill), dim=1).unsqueeze(1)


def test_actual_fourier_proposal_flows_through_solver_gate_to_accepted_incumbent():
    source = generate_unified_fourier_binary('source', master_seed=1, material_fraction_bin=1,
        config={'fourier_cutoff': 2, 'audit_resolution': 16})
    config = FullCellMpbBandSolverConfig(num_bands=2, resolution=16, polarization='TE',
        epsilon_air=source.epsilon_air, epsilon_material=source.epsilon_material, eigensolver_tolerance=1e-7)
    options = {'verification_resolution': 24, 'initial_spacing': .25, 'min_levels': 2,
               'max_levels': 2, 'required_stable_levels': 2, 'edge_tolerance': 1,
               'resolution_tolerance': 1e-6}
    points = np.array([[0, 0], [.25, 0]])
    def solver(item, *, k_points, config):
        fill = float(item.rasterize((16, 16)).mean())
        return np.tile([.2+.1*fill, .8-.1*fill], (len(k_points), 1)), {'geometry_hash': item.geometry_hash}
    def evaluate(item):
        return evaluate_feedback_candidate(item, k_points=points, config=config, lower_band=1,
                                           refinement=options, solver=solver)
    anchor = evaluate(source)
    assert anchor['status'] == 'numerically_stable'
    threshold = source.parameters['threshold']
    state = start_feedback(incumbent_hash=source.parameter_hash, incumbent_parameters=[threshold],
        incumbent_observation=anchor['observation'], target_frequency=.5, radius=.1,
        min_radius=.01, max_radius=.2, candidate_budget=1)
    candidates, raw_records, saved = {}, [], []
    def propose(current):
        result = propose_fourier_candidate(current, source, ToyPredictor(), coefficient_keys=(),
            global_bounds=((threshold-.5, threshold+.5),), origin=[0], scales=[1],
            max_mode_radius=1, seed=3, population_size=5, surrogate_budget=10,
            raster_method='master_block_mean_v1', source_resolution=16, input_resolution=8,
            material_fraction_bounds=(0, 1), lower_band=1, surrogate_contract_sha256='model')
        assert result['status'] == 'surrogate_proposal_only'
        candidates[result['proposal']['parameter_hash']] = FullCellStructureSpec.from_mapping(result['structure'])
        return result['proposal']
    def solver_attempt(pending):
        result = evaluate(candidates[pending.pending.parameter_hash])
        raw_records.append(result)
        return result['observation']
    final, status = run_feedback_loop(state, propose=propose, evaluate=solver_attempt, persist=saved.append)
    assert status == 'budget_exhausted'
    assert final.incumbent_hash != source.parameter_hash
    assert final.history[0]['accepted'] and raw_records[0]['status'] == 'numerically_stable'
    assert saved[0].pending is not None and saved[-1].pending is None
