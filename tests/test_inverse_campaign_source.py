"""Check scheduler source shards follow the registered manifest roster."""
import pytest
from phc2d.inverse.campaign_source import source_id_for_shard


def test_source_shard_uses_manifest_order_and_rejects_invalid_roster():
    manifest = {'sources': [{'source_id': 51}, {'source_id': 8}, {'source_id': 77}]}
    assert source_id_for_shard(manifest, 1) == 8
    with pytest.raises(ValueError, match='roster'):
        source_id_for_shard(manifest, 3)
    with pytest.raises(ValueError, match='roster'):
        source_id_for_shard({'sources': [{'source_id': 8}, {'source_id': 8}]}, 0)
