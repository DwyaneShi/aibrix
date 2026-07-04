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

// Builds the ResourceAllocation projection for the TCE backend. Missing fields
// are tolerated and simply omitted.

package impl

import (
	"strings"

	plannerclient "github.com/vllm-project/aibrix/apps/console/api/planner/client"
	rmtypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
	"k8s.io/klog/v2"
)

// defaultRoleName matches the StormService default role.
const defaultRoleName = "default"

// buildTCEResourceAllocation projects a TCE ProvisionResult into the
// extra_body.aibrix.resource_allocation payload. Provider-specific details are
// mapped best-effort; missing fields are omitted. Replicas come from the RM
// response when present, otherwise they fall back to the requested replicas.
func buildTCEResourceAllocation(prov *rmtypes.ProvisionResult, fallbackReplicas int) *plannerclient.TCEResourceAllocation {
	if fallbackReplicas <= 0 {
		fallbackReplicas = defaultJobReplicas
	}
	allocation := &plannerclient.TCEResourceAllocation{
		ProvisionID: prov.ProvisionID,
	}

	details := &plannerclient.ResourceDetails{
		Provider: string(rmtypes.ResourceProvisionTypeTCE),
	}

	resource := plannerclient.ResourceItem{
		Name:                defaultRoleName,
		Replica:             fallbackReplicas,
		AcceleratorCategory: "gpu",
	}

	populateTCEDetails(details, &resource, prov)

	// Emit the resource entry only when it carries an accelerator request.
	if resource.AcceleratorType != "" || resource.AcceleratorCount != 0 {
		details.Resources = []plannerclient.ResourceItem{resource}
	}
	details.Replica = resource.Replica

	allocation.ResourceDetails = []plannerclient.ResourceDetails{*details}
	return allocation
}

func replicasFromProvisionSpec(spec rmtypes.ResourceProvisionSpec) int {
	if spec.Groups == nil || len(*spec.Groups) == 0 {
		return defaultJobReplicas
	}
	replicas := (*spec.Groups)[0].Replicas
	if replicas == nil || *replicas <= 0 {
		return defaultJobReplicas
	}
	return *replicas
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

	details.EndpointCluster = seg.Region.String()
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

	// Adjust the accelerator count to be the number of accelerators per replica.
	if resource.Replica > 0 {
		if resource.AcceleratorCount%resource.Replica != 0 {
			klog.Errorf("TCE allocation segment has accelerator count %d that is not divisible by replica count %d", resource.AcceleratorCount, resource.Replica)
		}
		resource.AcceleratorCount = resource.AcceleratorCount / resource.Replica
	}

	// Ensure the accelerator count is at least 1.
	if resource.AcceleratorCount <= 0 {
		resource.AcceleratorCount = 1
	}

	// Mirror onto the flat fields the console handler/UI read; only set on the
	// real allocation path so the card stays blank when there's no segment.
	details.GpuType = resource.AcceleratorType
	details.Replica = resource.Replica
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
