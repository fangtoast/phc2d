"""Shared HPC execution contracts for MPB and Meep physical workloads.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from phc2d.hpc.adapters import PhysicalAdapterSpec, get_adapter_spec, list_adapter_specs
from phc2d.hpc.contracts import (
    CAMPAIGN_STATES,
    PhysicalCampaignManifest,
    PhysicalCase,
    PhysicalPlanManifest,
    PhysicalResultRecord,
    PhysicalShardSpec,
    atomic_write_json,
    sha256_file,
    sha256_payload,
    validate_shard_coverage,
)
from phc2d.hpc.resources import (
    BenchmarkEvidence,
    PlatformProfile,
    admitted_scaling_tiers,
    benchmark_evidence_from_partition,
    load_platform_profile,
    recommend_node_count,
)
from phc2d.hpc.campaign import STANDARD_FINAL_ARTIFACTS, audit_campaign, collect_campaign_status
from phc2d.hpc.runtime import AdapterNotMigratedError, PhysicalTaskAdapter, get_runtime_adapter

__all__ = [
    "BenchmarkEvidence",
    "AdapterNotMigratedError",
    "CAMPAIGN_STATES",
    "PhysicalAdapterSpec",
    "PhysicalCampaignManifest",
    "PhysicalCase",
    "PhysicalPlanManifest",
    "PhysicalResultRecord",
    "PhysicalShardSpec",
    "PhysicalTaskAdapter",
    "PlatformProfile",
    "STANDARD_FINAL_ARTIFACTS",
    "admitted_scaling_tiers",
    "benchmark_evidence_from_partition",
    "atomic_write_json",
    "audit_campaign",
    "collect_campaign_status",
    "get_adapter_spec",
    "get_runtime_adapter",
    "list_adapter_specs",
    "load_platform_profile",
    "recommend_node_count",
    "sha256_file",
    "sha256_payload",
    "validate_shard_coverage",
]
