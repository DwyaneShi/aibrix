/*
Copyright 2026 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Builds the TCEDecision projection for the TCE backend. Missing fields are
// tolerated and simply omitted.

package impl

import (
	"os"
	"strings"
	"time"

	plannerclient "github.com/vllm-project/aibrix/apps/console/api/planner/client"
	rmtypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
)

// defaultRoleName matches the StormService default role.
const defaultRoleName = "default"

// buildTCEDecision projects a TCE ProvisionResult into the TCEDecision
// payload that rides on extra_body.aibrix.planner_decision. Provider-specific
// details are mapped best-effort; missing fields are omitted. Replicas
// defaults to 1 (single-replica only today).
func buildTCEDecision(
	prov *rmtypes.ProvisionResult,
	gpuType string,
	gpusPerReplica int,
) *plannerclient.TCEDecision {
	dec := &plannerclient.TCEDecision{
		ProvisionID: prov.ProvisionID,
	}

	details := &plannerclient.ResourceDetails{
		Provider: string(rmtypes.ResourceProvisionTypeTCE),
	}

	resource := plannerclient.ResourceItem{
		Name:                defaultRoleName,
		Replica:             1,
		AcceleratorCategory: "gpu",
	}
	if !strings.EqualFold(gpuType, "CPU") {
		resource.AcceleratorType = gpuType
	}
	resource.AcceleratorCount = gpusPerReplica

	populateTCEDetails(details, &resource, prov)

	// Emit the resource entry only when it carries an accelerator request.
	if resource.AcceleratorType != "" || resource.AcceleratorCount != 0 {
		details.Resources = []plannerclient.ResourceItem{resource}
	}

	dec.ResourceDetails = details
	return dec
}

func populateTCEDetails(details *plannerclient.ResourceDetails, resource *plannerclient.ResourceItem, prov *rmtypes.ProvisionResult) {
	details.SaleMode = "scheduled"
	details.QoS = "shared_cores"

	// Demo mode: use hardcoded fields instead of deriving from prov.
	if isTCEDemoMode() {
		applyTCEDemoOverrides(details, resource)
		return
	}

	if prov.TCE == nil || prov.TCE.GroupResults == nil {
		return
	}
	groups := *prov.TCE.GroupResults
	if len(groups) == 0 || len(groups[0].AllocationSegments) == 0 {
		return
	}
	seg := groups[0].AllocationSegments[0]

	details.EndpointCluster = seg.Region.PhysicalCluster
	if seg.Region.PhysicalCluster != "" && seg.Region.Dc != "" {
		details.EndpointCluster = seg.Region.PhysicalCluster + "-" + seg.Region.Dc
	}
	details.LogicalCluster = seg.Region.LogicalCluster
	if seg.CommitInfo != nil && seg.CommitInfo.ResourcePoolName != nil {
		details.ResourcePoolName = *seg.CommitInfo.ResourcePoolName
	}

	if seg.AcceleratorType != "" {
		resource.AcceleratorType = seg.AcceleratorType
	}
	if seg.AcceleratorCategory != "" {
		resource.AcceleratorCategory = seg.AcceleratorCategory
	}
	if seg.Count != nil && *seg.Count > 0 {
		resource.AcceleratorCount = *seg.Count
	}
	if seg.Replicas != nil && *seg.Replicas > 0 {
		resource.Replica = *seg.Replicas
	}
}

// isTCEDemoMode checks the AIBRIX_TCE_DEMO_MODE env toggle.
func isTCEDemoMode() bool {
	switch os.Getenv("AIBRIX_TCE_DEMO_MODE") {
	case "1", "true", "TRUE", "True", "yes", "YES", "Yes":
		return true
	}
	return false
}

// applyTCEDemoOverrides fills the MDS submission fields with hardcoded demo
// values. Edit them in place to point the demo at a different cluster/pool.
func applyTCEDemoOverrides(details *plannerclient.ResourceDetails, resource *plannerclient.ResourceItem) {
	details.SaleMode = "reserved"
	details.QoS = "shared_cores"
	details.EndpointCluster = "Echo-HL"
	details.LogicalCluster = "ai"
	details.ResourcePoolName = "compute-3530-hl-echo-ai-default"

	resource.Name = defaultRoleName
	resource.AcceleratorType = "NVIDIA-H20"
	resource.AcceleratorCategory = "gpu"
	resource.AcceleratorCount = 1
	resource.Replica = 1
}

// newFakeTCEProvisionResult fabricates a "running" ProvisionResult for demo
// mode so the worker can skip RM provisioning and proceed to CreateBatch.
func newFakeTCEProvisionResult(jobID string) *rmtypes.ProvisionResult {
	now := time.Now()
	return &rmtypes.ProvisionResult{
		ProvisionID:    strings.TrimPrefix(jobID, "job_"),
		IdempotencyKey: jobID,
		Status:         rmtypes.ProvisionStatusRunning,
		CreatedAt:      now,
		UpdatedAt:      now,
		ExtensionProvisionResultDetails: rmtypes.ExtensionProvisionResultDetails{
			TCE: &rmtypes.TCEProvisionDetail{},
		},
	}
}
