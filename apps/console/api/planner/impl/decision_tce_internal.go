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
	"strings"

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

	dec.ResourceDetails = []plannerclient.ResourceDetails{*details}
	return dec
}

func populateTCEDetails(details *plannerclient.ResourceDetails, resource *plannerclient.ResourceItem, prov *rmtypes.ProvisionResult) {
	details.SaleMode = "scheduled"
	details.QoSLevel = "shared_cores"

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
		details.ResourcePoolName = trimResourcePoolSuffix(*seg.CommitInfo.ResourcePoolName)
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

// trimResourcePoolSuffix drops the trailing priority class segment
// (e.g. "-guarantee") that the scheduling system appends to the pool
// name. The MDS queue-name expects the pool identifier without it,
// e.g. "compute-3530-hl-federationgpu-default-default-guarantee" ->
// "compute-3530-hl-federationgpu-default-default".
func trimResourcePoolSuffix(name string) string {
	if idx := strings.LastIndex(name, "-"); idx > 0 {
		return name[:idx]
	}
	return name
}
