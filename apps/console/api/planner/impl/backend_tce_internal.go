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
	"strings"
	"time"

	"k8s.io/klog/v2"

	plannerapi "github.com/vllm-project/aibrix/apps/console/api/planner/api"
	plannerclient "github.com/vllm-project/aibrix/apps/console/api/planner/client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provisioner"
	rmtypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
	"github.com/vllm-project/aibrix/apps/console/api/utils"
)

var tceMatchingAcceleratorTypeAliases = map[string]string{
	"NVIDIA-A100-SXM4-80GB": "A100-SXM-80GB",
	"A100-SXM4-80GB":        "A100-SXM-80GB",
}

const (
	tceRTX6000DAcceleratorType = "NVIDIA-RTX-6000D"
	tceRTX6000DCPUCoresPerGPU  = 14
	tceResourceTypeScheduled   = "scheduled"
)

var tceMatchingCPUCoresPerGPU = map[string]int{
	tceRTX6000DAcceleratorType: tceRTX6000DCPUCoresPerGPU,
}

func init() {
	RegisterBackend(rmtypes.ResourceProvisionTypeTCE, func(provisioner.Provisioner) plannerBackend {
		return &tcePlannerBackend{}
	})
	RegisterPlanningPolicy(rmtypes.ResourceProvisionTypeTCE, PlanningPolicyTypeSimple, func(cfg PolicyConfig) (PlanningPolicy[*queuedJob], error) {
		return &SimplePolicy{cfg}, nil
	})
}

type tcePlannerBackend struct{}

func (b *tcePlannerBackend) FormatRegion(region *rmtypes.RegionSpec) string {
	if region == nil || region.TCE == nil {
		return ""
	}
	return region.TCE.Dc
}

func (b *tcePlannerBackend) ValidateRequest(req *plannerapi.EnqueueRequest) error {
	if req.ModelTemplate == nil || req.ModelTemplate.Name == "" {
		return fmt.Errorf("%w: missing model_template", plannerapi.ErrInvalidJob)
	}
	completionWindow, err := utils.ParseCompletionWindow(string(req.BatchParams.CompletionWindow))
	if err != nil {
		return fmt.Errorf("%w: parse completion window: %v", plannerapi.ErrInvalidJob, err)
	}
	matchingWindow := max(completionWindow, time.Hour)
	providerConfig := tceProviderConfig(req)
	resourceType, resourceTypeConfigured, err := parseTCEProviderResourceType(providerConfig)
	if err != nil {
		return fmt.Errorf("%w: %v", plannerapi.ErrInvalidJob, err)
	}
	if _, err := parseTCEProviderPSM(
		providerConfig,
		resourceTypeConfigured && resourceType == tceResourceTypeScheduled,
	); err != nil {
		return fmt.Errorf("%w: %v", plannerapi.ErrInvalidJob, err)
	}
	if _, err := parseTCEProviderRegions(providerConfig); err != nil {
		return fmt.Errorf("%w: %v", plannerapi.ErrInvalidJob, err)
	}
	if _, err := parseTCEProviderDuration(providerConfig, matchingWindow); err != nil {
		return fmt.Errorf("%w: %v", plannerapi.ErrInvalidJob, err)
	}
	return nil
}

func (b *tcePlannerBackend) Schedule(_ context.Context, req *plannerapi.EnqueueRequest) (spec rmtypes.ResourceProvisionSpec, err error) {
	gpuType, gpusPerReplica, err := decodeAcceleratorFromTemplate(req.ModelTemplate)
	if err != nil {
		return spec, err
	}
	if canonicalType, ok := tceMatchingAcceleratorTypeAliases[gpuType]; ok {
		gpuType = canonicalType
	}

	// Default and minimum matching window is 1 hour.
	matchingWindow := time.Hour
	startTime := time.Now().UTC().Add(5 * time.Minute)
	if req.BatchParams.CompletionWindow != "" {
		completionWindow, parseErr := utils.ParseCompletionWindow(string(req.BatchParams.CompletionWindow))
		if parseErr != nil {
			return spec, fmt.Errorf("parse completion window: %w", parseErr)
		}
		if completionWindow > matchingWindow {
			matchingWindow = completionWindow
		}
	}
	endTime := startTime.Add(matchingWindow)

	providerConfig := tceProviderConfig(req)
	resourceType, resourceTypeConfigured, err := parseTCEProviderResourceType(providerConfig)
	if err != nil {
		return spec, err
	}
	exactDurationHours, err := parseTCEProviderDuration(providerConfig, matchingWindow)
	if err != nil {
		return spec, err
	}
	psm, err := parseTCEProviderPSM(
		providerConfig,
		resourceTypeConfigured && resourceType == tceResourceTypeScheduled,
	)
	if err != nil {
		return spec, err
	}
	regions, err := parseTCEProviderRegions(providerConfig)
	if err != nil {
		return spec, err
	}

	group := buildProvisionGroupPlan(gpuType, gpusPerReplica, requestedReplicas(req))
	if len(regions) > 0 {
		group.TCE = &rmtypes.TCEGroupOptions{}
		group.TCE.RegionAffinity.Dc.Required = &regions
	}
	if cpuCoresPerGPU, ok := tceMatchingCPUCoresPerGPU[gpuType]; ok && gpusPerReplica > 0 {
		cpuCoresPerReplica := cpuCoresPerGPU * gpusPerReplica
		group.CpuCoresPerReplica = &cpuCoresPerReplica
	}

	tceCredential := &rmtypes.TCECredential{}
	if psm != "" {
		tceCredential.PSM = &psm
	}
	spec = rmtypes.ResourceProvisionSpec{
		Credential: rmtypes.ResourceCredential{
			Provider: rmtypes.ResourceProvisionTypeTCE,
			ExtensionResourceCredentials: rmtypes.ExtensionResourceCredentials{
				TCE: tceCredential,
			},
		},
		Groups: &[]rmtypes.ResourceGroupSpec{group},
		TimeWindow: &rmtypes.TimeWindow{
			StartTime:   startTime,
			EndTime:     &endTime,
			MinDuration: exactDurationHours,
			MaxDuration: exactDurationHours,
		},
	}
	return spec, nil
}

func tceProviderConfig(req *plannerapi.EnqueueRequest) map[string]any {
	if req == nil || req.ResourceRequest == nil {
		return nil
	}
	return req.ResourceRequest.ProviderConfig
}

func parseTCEProviderResourceType(providerConfig map[string]any) (string, bool, error) {
	value, ok := providerConfig["resource_type"]
	if !ok {
		return tceResourceTypeScheduled, false, nil
	}
	resourceType, ok := value.(string)
	if !ok || strings.TrimSpace(resourceType) == "" {
		return "", true, fmt.Errorf("provider_config.resource_type must be a resource type string")
	}
	resourceType = strings.ToLower(strings.TrimSpace(resourceType))
	if resourceType != tceResourceTypeScheduled {
		return "", true, fmt.Errorf("provider_config.resource_type %q is not supported yet", resourceType)
	}
	return resourceType, true, nil
}

func parseTCEProviderPSM(providerConfig map[string]any, required bool) (string, error) {
	value, ok := providerConfig["psm"]
	if !ok {
		if required {
			return "", fmt.Errorf("provider_config.psm is required for scheduled resources")
		}
		return "", nil
	}
	psm, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("provider_config.psm must be a string")
	}
	psm = strings.TrimSpace(psm)
	if psm == "" && required {
		return "", fmt.Errorf("provider_config.psm is required for scheduled resources")
	}
	if psm == "" {
		return "", nil
	}
	return psm, nil
}

func parseTCEProviderRegions(providerConfig map[string]any) ([]string, error) {
	value, ok := providerConfig["regions"]
	if !ok || value == nil {
		return nil, nil
	}
	values, ok := value.([]any)
	if !ok {
		if typedRegions, ok := value.([]string); ok {
			values = make([]any, len(typedRegions))
			for i, region := range typedRegions {
				values[i] = region
			}
		} else {
			return nil, fmt.Errorf("provider_config.regions must be an array of strings")
		}
	}
	regions := make([]string, 0, len(values))
	for _, value := range values {
		region, ok := value.(string)
		region = strings.TrimSpace(region)
		if !ok || region == "" {
			return nil, fmt.Errorf("provider_config.regions must contain non-empty strings")
		}
		regions = append(regions, region)
	}
	return regions, nil
}

func parseTCEProviderDuration(
	providerConfig map[string]any,
	matchingWindow time.Duration,
) (*int, error) {
	value, ok := providerConfig["duration"]
	if !ok {
		return nil, nil
	}
	durationValue, ok := value.(string)
	if !ok || durationValue == "" {
		return nil, fmt.Errorf("provider_config.duration must be a duration string")
	}
	duration, err := utils.ParseCompletionWindow(durationValue)
	if err != nil {
		return nil, fmt.Errorf("parse provider_config.duration: %w", err)
	}
	if duration%time.Hour != 0 {
		return nil, fmt.Errorf("provider_config.duration must use whole hours")
	}
	if duration > matchingWindow {
		return nil, fmt.Errorf("provider_config.duration must not exceed completion window")
	}
	durationHours := int(duration / time.Hour)
	return &durationHours, nil
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
	if timeWindow := b.AllocationTimeWindow(prov); timeWindow != nil && timeWindow.EndTime != nil {
		allocation.ProvisionResourceDeadline = timeWindow.EndTime.Unix()
	}
	return allocation
}

func (b *tcePlannerBackend) AllocationTimeWindow(prov *rmtypes.ProvisionResult) *rmtypes.TimeWindow {
	if prov == nil || prov.TCE == nil || prov.TCE.GroupResults == nil {
		return nil
	}

	var (
		startTime time.Time
		endTime   *time.Time
	)
	for _, group := range *prov.TCE.GroupResults {
		for _, segment := range group.AllocationSegments {
			if !segment.Allocated {
				continue
			}
			window := segment.TimeWindow
			if window.StartTime.After(startTime) {
				startTime = window.StartTime
			}
			if window.EndTime != nil &&
				(endTime == nil || window.EndTime.Before(*endTime)) {
				segmentEndTime := *window.EndTime
				endTime = &segmentEndTime
			}
		}
	}
	if startTime.IsZero() && endTime == nil {
		return nil
	}
	return &rmtypes.TimeWindow{
		StartTime: startTime,
		EndTime:   endTime,
	}
}

func (b *tcePlannerBackend) BuildRuntime(req *plannerapi.EnqueueRequest, prov *rmtypes.ProvisionResult) (*plannerapi.RuntimeRef, error) {
	if req == nil {
		return nil, fmt.Errorf("missing enqueue request")
	}

	return &plannerapi.RuntimeRef{Target: "Octagram"}, nil
}
