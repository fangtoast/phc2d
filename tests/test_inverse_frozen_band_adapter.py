"""Test frozen checkpoint reconstruction and input gradients.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from phc2d.inverse.frozen_band_adapter import (FrozenBandAdapter, SingleSampleBudgetedPredictor,
                                              audit_singleton_inference_evidence)
from phc2d.models import build_model


def test_singleton_budget_counts_actual_calls_and_reuses_only_identical_input():
    class Predictor:
        def predict_tensor(self, inputs):
            return inputs[:, :1, :1].clone()
    predictor = SingleSampleBudgetedPredictor(Predictor(), max_inferences=2)
    first = torch.ones((1, 2, 2), dtype=torch.float32)
    result = predictor.predict_tensor(first)
    result.fill_(9)  # A caller cannot mutate the cached result.
    assert predictor.predict_tensor(first.clone()).item() == 1
    assert predictor.predict_tensor(first * 2).item() == 2
    assert (predictor.requests, predictor.inferences, predictor.cache_hits) == (3, 2, 1)
    with pytest.raises(ValueError, match='budget exhausted'):
        predictor.predict_tensor(first)
    with pytest.raises(ValueError, match='one non-gradient'):
        predictor.predict_tensor(torch.ones((2, 2, 2)))


def test_failed_inference_consumes_budget():
    class Failed:
        def predict_tensor(self, inputs):
            raise RuntimeError('model failed')
    predictor = SingleSampleBudgetedPredictor(Failed(), max_inferences=1)
    with pytest.raises(RuntimeError, match='model failed'):
        predictor.predict_tensor(torch.ones((1, 2, 2)))
    with pytest.raises(ValueError, match='budget exhausted'):
        predictor.predict_tensor(torch.ones((1, 2, 2)))


def test_bounded_cache_reuses_previous_input_and_evicts_least_recently_used():
    class Predictor:
        def predict_tensor(self, inputs):
            return inputs.clone()
    predictor = SingleSampleBudgetedPredictor(Predictor(), max_inferences=3, cache_capacity=2)
    one = torch.ones((1, 2, 2))
    two, three = 2 * one, 3 * one
    predictor.predict_tensor(one)
    predictor.predict_tensor(two)
    predictor.predict_tensor(one)
    predictor.predict_tensor(three)
    assert predictor.inferences == 3 and predictor.cache_hits == 1
    with pytest.raises(ValueError, match='budget exhausted'):
        predictor.predict_tensor(two)


def test_inference_start_evidence_survives_predictor_failure(tmp_path):
    import json
    class Failed:
        def predict_tensor(self, inputs):
            raise RuntimeError('interrupted')
    directory = tmp_path/'evidence'
    predictor = SingleSampleBudgetedPredictor(Failed(), max_inferences=1, evidence_directory=directory)
    with pytest.raises(RuntimeError, match='interrupted'):
        predictor.predict_tensor(torch.ones((1, 2, 2)))
    event = json.loads((directory/'request_0001.json').read_bytes())
    assert event['kind'] == 'inference_started' and event['inference'] == 1
    assert hashlib.sha256((directory/event['input_file']).read_bytes()).hexdigest() == event['input_file_sha256']
    assert not (directory/'completed_0001.json').exists()
    audit = audit_singleton_inference_evidence(directory, max_inferences=1)
    assert audit['status'] == 'unknown' and audit['unknown_inferences'] == [1]
    with pytest.raises(FileExistsError):
        SingleSampleBudgetedPredictor(Failed(), max_inferences=1, evidence_directory=directory)


def test_inference_evidence_distinguishes_completion_cache_and_denial(tmp_path):
    import json
    class Predictor:
        def predict_tensor(self, inputs):
            return inputs.clone()
    directory = tmp_path/'evidence'
    predictor = SingleSampleBudgetedPredictor(Predictor(), max_inferences=1, evidence_directory=directory)
    predictor.predict_tensor(torch.ones((1, 2, 2)))
    predictor.predict_tensor(torch.ones((1, 2, 2)))
    with pytest.raises(ValueError, match='budget exhausted'):
        predictor.predict_tensor(torch.zeros((1, 2, 2)))
    events = [json.loads((directory/f'request_{index:04d}.json').read_bytes()) for index in range(1, 4)]
    assert [event['kind'] for event in events] == ['inference_started', 'cache_hit', 'budget_exhausted']
    completed = json.loads((directory/'completed_0001.json').read_bytes())
    assert hashlib.sha256((directory/completed['output_file']).read_bytes()).hexdigest() == completed['output_file_sha256']
    audit = audit_singleton_inference_evidence(directory, max_inferences=1)
    assert audit['status'] == 'complete' and audit['completed'] == 1
    assert audit['cache_hits'] == 1 and audit['budget_denials'] == 1
    (directory/completed['output_file']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='output file mismatch'):
        audit_singleton_inference_evidence(directory, max_inferences=1)


def checkpoint(tmp_path):
    model = build_model("mlp", input_shape=(2,), output_shape=(2, 2))
    payload = dict(model_name="mlp", model_params={}, input_shape=(2,), output_shape=(2, 2),
                   state_dict=model.state_dict(), target_mean=torch.ones(2, 2),
                   target_std=torch.full((2, 2), 2.0))
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), model


def test_prediction_and_gradient(tmp_path):
    path, digest, model = checkpoint(tmp_path)
    model.eval()
    adapter = FrozenBandAdapter(path, sha256=digest)
    x = torch.tensor([[0.3, 0.4]], requires_grad=True)
    expected = model(x).reshape(1, 2, 2) * 2 + 1
    actual = adapter.predict_tensor(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(not p.requires_grad for p in adapter.model.parameters())


def test_rejects_wrong_hash_and_shape(tmp_path):
    path, digest, _ = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        FrozenBandAdapter(path, sha256="0" * 64)
    adapter = FrozenBandAdapter(path, sha256=digest)
    with pytest.raises(ValueError, match="input shape"):
        adapter.predict_tensor(torch.ones(1, 3))


def test_rejects_corrupt_k_identity(tmp_path):
    path, _, _ = checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["k_points"] = torch.zeros(2, 2)
    payload["k_points_sha256"] = "0" * 64
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="k-point SHA-256"):
        FrozenBandAdapter(path, sha256=digest)
