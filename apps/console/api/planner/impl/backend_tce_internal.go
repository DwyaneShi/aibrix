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

// Provides the TCE plannerBackend. It self-registers via init() into the
// backendRegistry defined in backend.go; newPlannerBackend resolves it for
// rmtypes.ResourceProvisionTypeTCE.

package impl

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/klog/v2"

	plannerapi "github.com/vllm-project/aibrix/apps/console/api/planner/api"
	plannerclient "github.com/vllm-project/aibrix/apps/console/api/planner/client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provisioner"
	rmtypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
)

func init() {
	RegisterBackend(rmtypes.ResourceProvisionTypeTCE, func(provisioner.Provisioner) plannerBackend {
		return &tcePlannerBackend{}
	})
	RegisterPlanningPolicy(rmtypes.ResourceProvisionTypeTCE, PlanningPolicyTypeSimple, func(cfg PolicyConfig) (PlanningPolicy[*queuedJob], error) {
		return &SimplePolicy{cfg}, nil
	})
}

type tcePlannerBackend struct{}

func (b *tcePlannerBackend) ValidateRequest(req *plannerapi.EnqueueRequest) error {
	if req.ModelTemplate == nil || req.ModelTemplate.Name == "" {
		return fmt.Errorf("%w: missing model_template", plannerapi.ErrInvalidJob)
	}
	return nil
}

func (b *tcePlannerBackend) Schedule(_ context.Context, req *plannerapi.EnqueueRequest) (spec rmtypes.ResourceProvisionSpec, err error) {
	gpuType, gpusPerReplica, err := decodeAcceleratorFromTemplate(req.ModelTemplate)
	if err != nil {
		return spec, err
	}

	// Default and minimum time window is 1 hour
	timeWindow := time.Hour
	startTime := time.Now().UTC().Add(5 * time.Minute)
	if req.BatchParams.CompletionWindow != "" {
		// Update time window if it's valid and greater than default time window
		if completionWindow, err := time.ParseDuration(string(req.BatchParams.CompletionWindow)); err == nil && completionWindow > timeWindow {
			timeWindow = completionWindow
		}
	}
	endTime := startTime.Add(timeWindow)

	spec = rmtypes.ResourceProvisionSpec{
		Credential: rmtypes.ResourceCredential{
			Provider: rmtypes.ResourceProvisionTypeTCE,
			ExtensionResourceCredentials: rmtypes.ExtensionResourceCredentials{
				TCE: &rmtypes.TCECredential{},
			},
		},
		Groups: &[]rmtypes.ResourceGroupSpec{buildProvisionGroupPlan(gpuType, gpusPerReplica, requestedReplicas(req))},
		TimeWindow: &rmtypes.TimeWindow{
			StartTime: startTime,
			EndTime:   &endTime,
		},
	}
	return spec, nil
}

func (b *tcePlannerBackend) LogProvisionResponse(jobID string, prov *rmtypes.ProvisionResult, spec rmtypes.ResourceProvisionSpec) {
	if prov == nil || prov.TCE == nil {
		return
	}
	klog.Infof("[planner] rm_response job_id=%q provision_id=%q status=%q provider=%q match_id=%q match_order_url=%q",
		jobID, prov.ProvisionID, prov.Status, spec.Credential.Provider, prov.TCE.MatchId, prov.TCE.MatchOrderUrl)
}

// logProvisionReady logs the TCE-specific detail of a ready provision.
// Folded out of plannerBackend; invoked by BuildResourceAllocation.
func (b *tcePlannerBackend) logProvisionReady(prov *rmtypes.ProvisionResult) {
	if prov == nil || prov.TCE == nil {
		return
	}
	var groupResults string
	if prov.TCE.GroupResults != nil {
		if payload, err := json.Marshal(prov.TCE.GroupResults); err == nil {
			groupResults = string(payload)
		} else {
			groupResults = fmt.Sprintf("<marshal error: %v>", err)
		}
	}
	klog.Infof("[planner] rm_ready provision_id=%q status=%q match_id=%q match_order_url=%q group_results=%s",
		prov.ProvisionID, prov.Status, prov.TCE.MatchId, prov.TCE.MatchOrderUrl, groupResults)
}

func (b *tcePlannerBackend) BuildResourceAllocation(spec rmtypes.ResourceProvisionSpec, prov *rmtypes.ProvisionResult) plannerclient.ResourceAllocation {
	b.logProvisionReady(prov)
	allocation := buildTCEResourceAllocation(prov, replicasFromProvisionSpec(spec))
	if spec.TimeWindow != nil && spec.TimeWindow.EndTime != nil {
		allocation.ProvisionResourceDeadline = spec.TimeWindow.EndTime.Unix()
	}
	return allocation
}

func (b *tcePlannerBackend) BuildRuntime(req *plannerapi.EnqueueRequest, prov *rmtypes.ProvisionResult) (*plannerapi.RuntimeRef, error) {
	if req == nil {
		return nil, fmt.Errorf("missing enqueue request")
	}

	return &plannerapi.RuntimeRef{Target: "Octagram"}, nil
}
