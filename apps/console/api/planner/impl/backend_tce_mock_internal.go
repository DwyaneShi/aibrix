/*
Copyright 2025 The Aibrix Team.

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

// TCE mock backend; used in demo deployments where a real TCE RM is not
// reachable. Selected automatically when the operator configures the
// resource manager with provisioner type "tceMock" — the planner factory
// looks up this backend by ResourceProvisionTypeTCEMock.
//
// Scope: this file owns demo-mode decision shaping. The matching mock
// provisioner (resource_manager/provisioner/tce_mock.go) returns a
// "running" ProvisionResult immediately so the planner's
// waitForProvisionReady poll succeeds on the first iteration. All
// hardcoded demo cluster/pool values live in this file so the production
// code paths (backend_tce_internal.go, decision_tce_internal.go) stay
// free of demo branching.

package impl

import (
	"context"
	"fmt"

	plannerapi "github.com/vllm-project/aibrix/apps/console/api/planner/api"
	plannerclient "github.com/vllm-project/aibrix/apps/console/api/planner/client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provisioner"
	rmtypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
)

func init() {
	RegisterBackend(rmtypes.ResourceProvisionTypeTCEMock, func(provisioner.Provisioner) plannerBackend {
		return &tceMockBackend{}
	})
}

// tceMockBackend embeds tcePlannerBackend so it inherits ValidateRequest
// and the LogProvisionResponse capability unchanged. It overrides Schedule
// to stamp the mock provider on the spec, and BuildResourceAllocation to substitute
// hardcoded demo cluster/pool values in place of fields that would
// normally be derived from a real RM ProvisionResult.
type tceMockBackend struct {
	tcePlannerBackend
}

// Schedule reuses tcePlannerBackend.Schedule but rewrites the credential
// provider so the request is routed through the matching tceMock
// provisioner rather than the real TCE one.
func (b *tceMockBackend) Schedule(ctx context.Context, req *plannerapi.EnqueueRequest) (spec rmtypes.ResourceProvisionSpec, gpuType string, gpusPerReplica int, err error) {
	spec, gpuType, gpusPerReplica, err = b.tcePlannerBackend.Schedule(ctx, req)
	if err != nil {
		return
	}
	spec.Credential.Provider = rmtypes.ResourceProvisionTypeTCEMock
	return
}

func (b *tceMockBackend) BuildResourceAllocation(spec rmtypes.ResourceProvisionSpec, prov *rmtypes.ProvisionResult, gpuType string, gpusPerReplica int) plannerclient.ResourceAllocation {
	allocation := buildTCEResourceAllocation(prov, gpuType, gpusPerReplica)
	if len(allocation.ResourceDetails) > 0 && len(allocation.ResourceDetails[0].Resources) > 0 {
		applyTCEDemoOverrides(&allocation.ResourceDetails[0], &allocation.ResourceDetails[0].Resources[0])
	} else {
		// Resources may have been omitted when buildTCEResourceAllocation saw no
		// accelerator request; demo mode always wants a populated entry.
		if len(allocation.ResourceDetails) == 0 {
			allocation.ResourceDetails = []plannerclient.ResourceDetails{{
				Provider: string(rmtypes.ResourceProvisionTypeTCE),
			}}
		}
		resource := plannerclient.ResourceItem{Name: defaultRoleName}
		applyTCEDemoOverrides(&allocation.ResourceDetails[0], &resource)
		allocation.ResourceDetails[0].Resources = []plannerclient.ResourceItem{resource}
	}
	if spec.TimeWindow != nil && spec.TimeWindow.EndTime != nil {
		allocation.ProvisionResourceDeadline = spec.TimeWindow.EndTime.Unix()
	}
	return allocation
}

func (b *tceMockBackend) BuildRuntime(req *plannerapi.EnqueueRequest, prov *rmtypes.ProvisionResult) (*plannerapi.RuntimeRef, error) {
	if req == nil {
		return nil, fmt.Errorf("missing enqueue request")
	}
	return &plannerapi.RuntimeRef{
		Target: "Octagram",
	}, nil
}

// applyTCEDemoOverrides fills the MDS submission fields with hardcoded demo
// values. Edit them in place to point the demo at a different cluster/pool.
func applyTCEDemoOverrides(details *plannerclient.ResourceDetails, resource *plannerclient.ResourceItem) {
	details.SaleMode = "reserved"
	details.QoSLevel = "shared_cores"
	details.EndpointCluster = "Echo-HL"
	details.LogicalCluster = "ai"
	details.ResourcePoolName = "compute-3530-hl-echo-ai-default"

	resource.Name = defaultRoleName
	resource.AcceleratorType = "NVIDIA-H20"
	resource.AcceleratorCategory = "gpu"
	resource.AcceleratorCount = 1
	resource.Replica = 1
}
