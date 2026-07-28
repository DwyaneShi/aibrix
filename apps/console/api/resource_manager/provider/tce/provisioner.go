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

package tce

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"
	"k8s.io/utils/lru"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/gpu_center_client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/scheduled_plan_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/utils"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provisioner"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
	"github.com/vllm-project/aibrix/apps/console/api/store"
)

const (
	businessLineInfoCacheSize = 1000
)

// tceProvisioner implements provisioner.Provisioner for TCE.
type tceProvisioner struct {
	clientset             *tceClientset
	store                 store.Store
	businessLineInfoCache *lru.Cache
	businessLineInfoMu    sync.RWMutex
}

// newProvisioner creates a new TCE provisioner.
func newProvisioner(s store.Store) (provisioner.Provisioner, error) {
	cache := lru.New(businessLineInfoCacheSize)

	clientset, err := newTCEClientset()
	if err != nil {
		return nil, err
	}

	return &tceProvisioner{
		clientset:             clientset,
		store:                 s,
		businessLineInfoCache: cache,
	}, nil
}

// NewTCEProvisioner creates a new TCE provisioner with the given clientset.
// This is exported for testing purposes.
func NewTCEProvisioner(s store.Store, clientset *tceClientset) (provisioner.Provisioner, error) {
	cache := lru.New(businessLineInfoCacheSize)

	return &tceProvisioner{
		clientset:             clientset,
		store:                 s,
		businessLineInfoCache: cache,
	}, nil
}

// Type returns the provisioner type.
func (p *tceProvisioner) Type() types.ResourceProvisionType {
	return types.ResourceProvisionTypeTCE
}

// Provision creates a matching intent in TCE and persists provision result.
func (p *tceProvisioner) Provision(ctx context.Context, req *types.ResourceProvision) (*types.ProvisionResult, error) {
	if req == nil {
		return nil, types.ErrInvalidArgs
	}

	if req.Spec.Credential.Provider == "" {
		req.Spec.Credential.Provider = types.ResourceProvisionTypeTCE
	}
	if req.Spec.Credential.Provider != types.ResourceProvisionTypeTCE || req.Spec.Credential.TCE == nil {
		return nil, types.ErrInvalidArgs
	}
	if req.IdempotencyKey == "" {
		return nil, types.ErrInvalidArgs
	}

	existing, err := p.store.GetProvisionByIdempotencyKey(ctx, req.IdempotencyKey)
	if err == nil && existing != nil {
		return existing, nil
	}
	if err != nil && status.Code(err) != codes.NotFound {
		return nil, fmt.Errorf("get provision by idempotency key: %w", err)
	}

	provisionID := uuid.New().String()

	intentReq, err := p.buildMatchingIntentRequest(ctx, req, req.IdempotencyKey, provisionID)
	if err != nil {
		return nil, err
	}

	matchingResult, err := p.clientset.ResourceManagerClient.CreateScheduledMatch(ctx, intentReq)
	if err != nil {
		return nil, fmt.Errorf("create scheduled match failed: %w", err)
	}
	if matchingResult == nil {
		return nil, fmt.Errorf("create scheduled match failed: empty result")
	}

	klog.Infof("create scheduled match success, matchId %s", matchingResult.MatchId)

	matchOrderUrl := fmt.Sprintf("%s/%s", p.clientset.RegionConfig.ScheduledMatchFE, matchingResult.MatchId)
	klog.Infof("detailed matching log %s", matchOrderUrl)

	now := time.Now().UTC()
	result := &types.ProvisionResult{
		ProvisionID:    provisionID,
		IdempotencyKey: req.IdempotencyKey,
		Provider:       string(p.Type()),
		Status:         types.ProvisionStatusProvisioning,
		Region:         types.RegionUnknown, // will be updated when provision is running
		CreatedAt:      now,
		UpdatedAt:      now,
		ExtensionProvisionResultDetails: types.ExtensionProvisionResultDetails{
			TCE: &types.TCEProvisionDetail{
				MatchId:       matchingResult.MatchId,
				MatchOrderUrl: matchOrderUrl,
				GroupResults:  toTCEGroupResults(matchingResult.GroupResults),
			},
		},
	}

	if err := p.store.UpsertProvision(ctx, result); err != nil {
		klog.Errorf("insert provision failed: %v", err)
		// Rollback the scheduled match
		_, merr := p.clientset.ResourceManagerClient.CancelScheduledMatch(ctx, matchingResult.MatchId)
		if merr != nil {
			klog.Errorf("cancel scheduled match failed for provision %s, matchId %s: %v", provisionID, matchingResult.MatchId, merr)
			klog.Errorf("detailed matching log %s", matchOrderUrl)
		}
		return nil, fmt.Errorf("insert provision: %w", err)
	}

	return result, nil
}

// Release marks the provision as released. If the provision is running or provisioning,
// it first sets status to releasing, cancels the scheduled match, then sets status to released.
func (p *tceProvisioner) Release(ctx context.Context, provisionID string) error {
	provision, err := p.store.GetProvision(ctx, provisionID)
	if err != nil {
		return fmt.Errorf("get provision: %w", err)
	}
	if provision == nil {
		return types.ErrProvisionNotFound
	}

	if provision.TCE == nil || provision.TCE.MatchId == "" {
		err = p.store.DeleteProvision(ctx, provisionID)
		if err != nil {
			klog.Errorf("delete provision failed: %v", err)
		}
		return nil
	}

	// If already released, nothing to do
	if provision.Status == types.ProvisionStatusReleased {
		return nil
	}

	if provision.TCE != nil && provision.TCE.MatchId != "" {
		// Set status to releasing
		if err := p.store.UpdateProvisionStatus(ctx, provisionID, types.ProvisionStatusReleasing); err != nil {
			return fmt.Errorf("update provision status to releasing: %w", err)
		}

		// Cancel the scheduled match
		_, err := p.clientset.ResourceManagerClient.CancelScheduledMatch(ctx, provision.TCE.MatchId)
		if err != nil {
			klog.Warningf("cancel scheduled match failed for provision %s, matchId %s: %v", provisionID, provision.TCE.MatchId, err)
			klog.Warningf("detailed matching log %s", provision.TCE.MatchOrderUrl)
			// Continue to mark as released even if cancel fails
		}
	}

	// Set status to released
	return p.store.UpdateProvisionStatus(ctx, provisionID, types.ProvisionStatusReleased)
}

// List retrieves provisions matching the given criteria and refreshes provisioning status from TCE.
func (p *tceProvisioner) List(ctx context.Context, opts *types.ListOptions) ([]*types.ProvisionResult, error) {
	if opts == nil {
		opts = &types.ListOptions{}
	}

	results, err := p.store.ListProvisions(ctx, opts)
	if err != nil {
		return nil, err
	}

	for _, result := range results {
		if result == nil {
			continue
		}

		if result.TCE == nil || result.TCE.MatchId == "" {
			err = p.store.DeleteProvision(ctx, result.ProvisionID)
			if err != nil {
				klog.Errorf("delete provision failed: %v", err)
			}
			continue
		}

		upsertProvision := false
		if result.TCE.TicketPriority == nil && p.clientset.GPUCenterClient != nil {
			if ticketID, parseErr := strconv.ParseInt(result.TCE.MatchId, 10, 64); parseErr == nil {
				if ticketPriority, tpErr := p.clientset.GPUCenterClient.GetTicketPriority(ctx, ticketID); tpErr == nil && ticketPriority != nil {
					result.TCE.TicketPriority = toTCETicketPriority(ticketID, ticketPriority)
					upsertProvision = true
				} else if tpErr != nil {
					klog.V(4).Infof("GetTicketPriority(%d) failed (non-fatal): %v", ticketID, tpErr)
				}
			}
		}

		// Fetch matching order timeline from ResourceManagerClient
		if result.TCE.MatchId != "" {
			prevTimeline := result.TCE.Timeline
			if timeline, tErr := p.clientset.ResourceManagerClient.GetMatchTimeline(ctx, result.TCE.MatchId); tErr == nil {
				result.TCE.Timeline = toTCETimelineEntries(timeline)
				if !reflect.DeepEqual(result.TCE.Timeline, prevTimeline) {
					upsertProvision = true
				}
			} else {
				klog.V(4).Infof("GetMatchTimeline(%s) failed (non-fatal): %v", result.TCE.MatchId, tErr)
			}
		}

		if result.Status == types.ProvisionStatusProvisioning {
			matchingResult, err := p.clientset.ResourceManagerClient.GetScheduledMatch(ctx, result.TCE.MatchId)
			if err != nil || matchingResult == nil {
				continue
			}
			if matchingResult.Status == scheduled_plan_types.MatchingResultStatusSuccess || matchingResult.Status == scheduled_plan_types.MatchingResultStatusPartial {
				result.Status = types.ProvisionStatusRunning
				result.TCE.GroupResults = toTCEGroupResults(matchingResult.GroupResults)

				// Update region from groupResults
				// Assume all allocation segments have the same region
				if result.TCE.GroupResults != nil && len(*result.TCE.GroupResults) > 0 {
					segments := (*result.TCE.GroupResults)[0].AllocationSegments
					if len(segments) > 0 {
						result.Region = segments[0].Region.String()
					}
				}

				klog.Infof("provision %s region %s, with matchId %s is running: %s", result.ProvisionID, result.Region, result.TCE.MatchId, matchingResult.Status)
				groupResuslts, err := json.Marshal(result.TCE.GroupResults)
				if err != nil {
					klog.Errorf("failed to marshal groupResults to json: %v", err)
				} else {
					klog.Infof("groupResults: %s", groupResuslts)
				}
				klog.Infof("detailed matching log %s/%s", p.clientset.RegionConfig.ScheduledMatchFE, result.TCE.MatchId)
				upsertProvision = true
			} else if matchingResult.Status == scheduled_plan_types.MatchingResultStatusCancelling || matchingResult.Status == scheduled_plan_types.MatchingResultStatusCancelled {
				result.Status = types.ProvisionStatusReleased
				if !upsertProvision {
					if err := p.store.UpdateProvisionStatus(ctx, result.ProvisionID, types.ProvisionStatusReleased); err != nil {
						continue
					}
				}
				klog.Infof("provision %s region %s, with matchId %s is released: %v", result.ProvisionID, result.Region, result.TCE.MatchId, result.TCE.GroupResults)
				klog.Infof("detailed matching log %s/%s", p.clientset.RegionConfig.ScheduledMatchFE, result.TCE.MatchId)
			} else if matchingResult.Status == scheduled_plan_types.MatchingResultStatusFailed {
				result.Status = types.ProvisionStatusFailed
				var failedReason string
				if matchingResult.Explanation != nil {
					failedReason = *matchingResult.Explanation
					result.ErrorMessage = *matchingResult.Explanation
				} else {
					failedReason = "unknown"
				}
				if !upsertProvision {
					if err := p.store.UpdateProvisionStatus(ctx, result.ProvisionID, types.ProvisionStatusFailed); err != nil {
						continue
					}
				}
				klog.Warningf("provision %s region %s, with matchId %s is failed: %s", result.ProvisionID, result.Region, result.TCE.MatchId, failedReason)
				klog.Warningf("detailed matching log %s/%s", p.clientset.RegionConfig.ScheduledMatchFE, result.TCE.MatchId)
			}
		} else if result.Status == types.ProvisionStatusReleasing {
			// Cancel the scheduled match
			_, err := p.clientset.ResourceManagerClient.CancelScheduledMatch(ctx, result.TCE.MatchId)
			if err != nil {
				klog.Warningf("cancel scheduled match failed for provision %s, matchId %s: %v", result.ProvisionID, result.TCE.MatchId, err)
			}
			result.Status = types.ProvisionStatusReleased
			if !upsertProvision {
				if err := p.store.UpdateProvisionStatus(ctx, result.ProvisionID, types.ProvisionStatusReleased); err != nil {
					continue
				}
			}
			klog.Infof("provision %s region %s, with matchId %s is released: %v", result.ProvisionID, result.Region, result.TCE.MatchId, result.TCE.GroupResults)
			klog.Infof("detailed matching log %s/%s", p.clientset.RegionConfig.ScheduledMatchFE, result.TCE.MatchId)
		}

		if upsertProvision {
			if err := p.store.UpsertProvision(ctx, result); err != nil {
				klog.Errorf("update provision failed, provision %s, matchId %s: %v, will try again in next list", result.ProvisionID, result.TCE.MatchId, err)
			} else {
				klog.V(4).Infof("provision %s upserted, matchId %s", result.ProvisionID, result.TCE.MatchId)
			}
		}
	}

	return results, nil
}

// getCachedBusinessLineInfo retrieves BusinessLineInfo from cache or fetches it from ByteQuota API.
func (p *tceProvisioner) getCachedBusinessLineInfo(ctx context.Context, psm string) (*bytequota_client.BusinessLineInfo, error) {
	cacheKey := psm

	// Check cache first
	p.businessLineInfoMu.RLock()
	if cached, ok := p.businessLineInfoCache.Get(cacheKey); ok {
		p.businessLineInfoMu.RUnlock()
		klog.V(4).Infof("cache hit for business line info: psm=%s", psm)
		return cached.(*bytequota_client.BusinessLineInfo), nil
	}
	p.businessLineInfoMu.RUnlock()

	// Fetch from ByteQuota API
	klog.V(4).Infof("fetching business line info: psm=%s", psm)
	info, err := p.clientset.ByteQuotaClient.GetBusinessLineInfo(ctx, psm, AIBrixPlatformBusinessLinePlatform)
	if err != nil {
		return nil, fmt.Errorf("get business line info: %w", err)
	}

	// Cache the result
	p.businessLineInfoMu.Lock()
	p.businessLineInfoCache.Add(cacheKey, info)
	p.businessLineInfoMu.Unlock()

	return info, nil
}

func (p *tceProvisioner) buildMatchingIntentRequest(ctx context.Context, req *types.ResourceProvision, idempotencyKey, provisionID string) (*scheduled_plan_types.MatchingIntentRequest, error) {
	if req == nil || req.Spec.Credential.TCE == nil {
		return nil, fmt.Errorf("invalid args: TCE credential is required")
	}

	if req.Spec.TimeWindow == nil {
		return nil, fmt.Errorf("invalid args: time window is required")
	}

	groups, err := buildMatchingGroups(req.Spec.Groups)
	if err != nil {
		return nil, err
	}

	tceCredential := req.Spec.Credential.TCE

	// Get PSM for business line lookup
	psm := AIBrixPlatformPSM
	if tceCredential.PSM != nil && *tceCredential.PSM != "" {
		psm = *tceCredential.PSM
	}

	// Get business line info from cache or fetch it
	businessLineInfo, err := p.getCachedBusinessLineInfo(ctx, psm)
	if err != nil {
		klog.Warningf("failed to get business line info for psm=%s: %v, using empty values", psm, err)
		return nil, err
	}

	decisionDeadline := req.Spec.TimeWindow.StartTime.UnixMilli()
	klog.Infof("using decisionDeadline for provision %s: %d (%s)", provisionID,
		decisionDeadline, time.UnixMilli(decisionDeadline).Format(time.RFC3339))

	intent := &scheduled_plan_types.MatchingIntent{
		Groups: groups,
		Requester: &scheduled_plan_types.Requester{
			BusinessLineId:   businessLineInfo.BusinessLineId,
			BusinessLineName: businessLineInfo.BusinessLineName,
			ResourceGroupId:  businessLineInfo.ResourceGroupId,
			Platform:         scheduled_plan_types.RequesterPlatform(AIBrixPlatformName),
		},
		Workload: &scheduled_plan_types.Workload{
			Scene:    scheduled_plan_types.Serving,
			Priority: 60,
		},
		// CommitDealine==0, 撮合成功会立即从 Booked 转移到 Commiting 状态, 无需自己调用 ByteQuota 的 increase 接口
		CommitDeadline: utils.ToPtr[int64](0),
		// 最晚决策时间点, 超时后系统会返回当前最优结果
		DecisionDeadline: &decisionDeadline,
		ExtraFields: utils.ToPtr(map[string]interface{}{
			"idempotencyKey": idempotencyKey,
			"provisionID":    provisionID,
			"provisionPSM":   psm,
		}),
	}

	intent.TimeWindow = &scheduled_plan_types.TimeWindow{
		StartTime:          req.Spec.TimeWindow.StartTime,
		EndTime:            req.Spec.TimeWindow.EndTime,
		Timezone:           utils.ToPtr("UTC"),
		FlexibleAllocation: &scheduled_plan_types.FlexibleAllocation{},
	}

	return &scheduled_plan_types.MatchingIntentRequest{
		Name:           "aibrix-provision-" + provisionID,
		Description:    "Create a new provision through AIBrix",
		MatchingIntent: intent,
		IdempotencyKey: req.IdempotencyKey,
	}, nil
}

func buildMatchingGroups(groups *[]types.ResourceGroupSpec) (*[]scheduled_plan_types.GroupSpec, error) {
	if groups == nil || len(*groups) == 0 {
		return nil, fmt.Errorf("invalid args: groups is required")
	}

	matchingGroups := make([]scheduled_plan_types.GroupSpec, 0, len(*groups))
	for _, group := range *groups {
		affinityPolicy, err := toMatchingAffinityPolicy(group.TCE)
		if err != nil {
			return nil, err
		}

		matchingGroup := scheduled_plan_types.GroupSpec{
			GpusPerReplica:        group.GpusPerReplica,
			ReplicaAffinity:       affinityPolicy,
			AcceleratorPreference: toMatchingAcceleratorPreference(group.AcceleratorPreference),
		}
		if group.Replicas != nil {
			matchingGroup.Replicas = group.Replicas
		}
		if group.CpuCoresPerReplica != nil {
			matchingGroup.CpuCores = group.CpuCoresPerReplica
		}
		if group.GroupRole != nil {
			matchingGroup.GroupRole = group.GroupRole
		}
		if group.Network != nil {
			matchingGroup.Network = toMatchingNetwork(group.Network, group.Storage)
		}
		if group.TCE == nil {
			group.TCE = &types.TCEGroupOptions{}
		}
		matchingGroup.LocationConstraint = toMatchingLocationConstraint(&group.TCE.RegionAffinity)
		if group.TCE.GroupAffinity != nil {
			groupAffinity, err := toMatchingAffinityPolicyFromPolicies(group.TCE.GroupAffinity)
			if err != nil {
				return nil, err
			}
			matchingGroup.GroupAffinity = &groupAffinity
		}
		if group.TCE.TopologyConstraint != nil {
			matchingGroup.TopologyConstraint = group.TCE.TopologyConstraint
		}
		if group.TCE.NumaConfig != nil {
			matchingGroup.NumaConfig = &scheduled_plan_types.NUMAConfig{
				CpuPinning:                group.TCE.NumaConfig.CpuPinning,
				NumaAware:                 group.TCE.NumaConfig.NumaAware,
				NumaLocalMemoryGB:         group.TCE.NumaConfig.NumaLocalMemoryGB,
				NumaNodeCount:             group.TCE.NumaConfig.NumaNodeCount,
				NumaOptimizedInterconnect: group.TCE.NumaConfig.NumaOptimizedInterconnect,
				NumaRequired:              group.TCE.NumaConfig.NumaRequired,
			}
		}

		matchingGroups = append(matchingGroups, matchingGroup)
	}

	return &matchingGroups, nil
}

func toMatchingAcceleratorPreference(pref *types.AcceleratorPreference) scheduled_plan_types.AcceleratorPreference {
	result := scheduled_plan_types.AcceleratorPreference{}
	if pref == nil {
		return result
	}

	result.MinBandwidthGBps = pref.MinBandwidthGBps
	result.MinMemoryGB = pref.MinMemoryGB
	result.PreferHighBandwidth = pref.PreferHighBandwidth
	result.PreferredTypes = pref.PreferredTypes
	result.Weight = pref.Weight

	if pref.Advanced != nil {
		result.Advanced = &scheduled_plan_types.AcceleratorPreferenceAdvanced{
			PcieGen:                pref.Advanced.PcieGen,
			PcieLanes:              pref.Advanced.PcieLanes,
			VendorSpecificFeatures: pref.Advanced.VendorSpecificFeatures,
		}
	}

	if pref.PrecisionSupport != nil {
		precision := &scheduled_plan_types.AcceleratorPreferencePrecisionSupport{}
		if pref.PrecisionSupport.Required != nil {
			required := make([]scheduled_plan_types.AcceleratorPreferencePrecisionSupportRequired, 0, len(*pref.PrecisionSupport.Required))
			for _, value := range *pref.PrecisionSupport.Required {
				required = append(required, scheduled_plan_types.AcceleratorPreferencePrecisionSupportRequired(value))
			}
			precision.Required = &required
		}
		if pref.PrecisionSupport.Preferred != nil {
			preferred := make([]scheduled_plan_types.AcceleratorPreferencePrecisionSupportPreferred, 0, len(*pref.PrecisionSupport.Preferred))
			for _, value := range *pref.PrecisionSupport.Preferred {
				preferred = append(preferred, scheduled_plan_types.AcceleratorPreferencePrecisionSupportPreferred(value))
			}
			precision.Preferred = &preferred
		}
		result.PrecisionSupport = precision
	}

	return result
}

func toMatchingNetwork(network *types.GroupSpecNetwork, storageItems *[]string) *scheduled_plan_types.GroupSpecNetwork {
	if network == nil {
		return nil
	}

	result := &scheduled_plan_types.GroupSpecNetwork{
		MaxHops:          network.MaxHops,
		MinBandwidthGbps: network.MinBandwidthGbps,
	}
	if network.Rdma != nil {
		rdma := scheduled_plan_types.GroupSpecNetworkRdma(*network.Rdma)
		result.Rdma = &rdma
	}
	if storageItems != nil {
		storage := &scheduled_plan_types.GroupSpecNetworkStorageConnectivity{}
		for _, item := range *storageItems {
			switch item {
			case "byteNaS":
				v := true
				storage.ByteNaS = &v
			case "hdfs":
				v := true
				storage.Hdfs = &v
			default:
				if storage.Other == nil {
					other := make([]string, 0, len(*storageItems))
					storage.Other = &other
				}
				*storage.Other = append(*storage.Other, item)
			}
		}
		result.StorageConnectivity = storage
	}
	return result
}

func toMatchingAffinityPolicy(tceOptions *types.TCEGroupOptions) (scheduled_plan_types.AffinityPolicy, error) {
	if tceOptions == nil {
		return scheduled_plan_types.AffinityPolicy{Policies: []scheduled_plan_types.AffinityPolicyPolicies{scheduled_plan_types.SameSwitchS2}}, nil
	}
	return toMatchingAffinityPolicyFromPolicies(tceOptions.ReplicaAffinity)
}

func toMatchingAffinityPolicyFromPolicies(policies *types.AffinityPolicies) (scheduled_plan_types.AffinityPolicy, error) {
	if policies == nil || len(policies.Policies) == 0 {
		return scheduled_plan_types.AffinityPolicy{Policies: []scheduled_plan_types.AffinityPolicyPolicies{scheduled_plan_types.SameSwitchS2}}, nil
	}

	mappedPolicies := make([]scheduled_plan_types.AffinityPolicyPolicies, 0, len(policies.Policies))
	for _, policy := range policies.Policies {
		mapped, err := mapAffinityPolicy(policy)
		if err != nil {
			return scheduled_plan_types.AffinityPolicy{}, err
		}
		mappedPolicies = append(mappedPolicies, mapped)
	}
	return scheduled_plan_types.AffinityPolicy{Policies: mappedPolicies}, nil
}

func mapAffinityPolicy(policy types.AffinityPolicy) (scheduled_plan_types.AffinityPolicyPolicies, error) {
	switch policy {
	case types.AffinityPolicyNuma:
		return scheduled_plan_types.SameNumaNode, nil
	case types.AffinityPolicyHost:
		return scheduled_plan_types.SingleHost, nil
	case types.AffinityPolicyTor:
		return scheduled_plan_types.SameSwitchS0, nil
	case types.AffinityPolicyMinipod:
		return scheduled_plan_types.SameSwitchS1, nil
	case types.AffinityPolicyBigpod:
		return scheduled_plan_types.SameSwitchS2, nil
	default:
		return "", fmt.Errorf("unsupported affinity policy: %s", string(policy))
	}
}

func toMatchingLocationConstraint(regionAffinity *types.TCERegionAffinity) *scheduled_plan_types.LocationConstraint {
	if regionAffinity == nil {
		return nil
	}
	return &scheduled_plan_types.LocationConstraint{
		Zone:    toLocationAffinity(&regionAffinity.Zone),
		Dc:      toLocationAffinity(&regionAffinity.Dc),
		Cluster: toLocationAffinity(&regionAffinity.Cluster),
	}
}

func toLocationAffinity(regionAffinity *types.RegionAffinity) *scheduled_plan_types.LocationAffinity {
	if regionAffinity == nil {
		return nil
	}
	return &scheduled_plan_types.LocationAffinity{
		Forbidden: regionAffinity.Forbidden,
		Preferred: regionAffinity.Preferred,
		Required:  regionAffinity.Required,
	}
}

func toTCEGroupResults(groupResults *scheduled_plan_types.GroupResults) *types.TCEGroupResults {
	if groupResults == nil {
		return nil
	}

	results := make(types.TCEGroupResults, 0, len(*groupResults))
	for _, gr := range *groupResults {
		result := types.TCEGroupResult{
			AllocationSegments: toTCEAllocationSegments(gr.AllocationSegments),
		}
		results = append(results, result)
	}

	return &results
}

func toTCEAllocationSegments(segments []scheduled_plan_types.AllocationSegment) []types.TCEAllocationSegment {
	if segments == nil {
		return nil
	}

	results := make([]types.TCEAllocationSegment, 0, len(segments))
	for _, seg := range segments {
		result := types.TCEAllocationSegment{
			Id:                  seg.Id,
			Allocated:           seg.Allocated,
			Region:              toTCERegion(seg.Cluster),
			AcceleratorType:     seg.AcceleratorType,
			AcceleratorCategory: seg.AcceleratorCategory,
			Count:               seg.Count,
			Replicas:            seg.Replicas,
			NodeIds:             seg.NodeIds,
			AcceleratorIds:      seg.AcceleratorIds,
			TimeWindow:          toTCETimeWindow(seg.TimeWindow),
			Preemptible:         seg.Preemptible,
			CommitInfo:          toTCECommitInfo(seg.CommitInfo),
		}
		results = append(results, result)
	}

	return results
}

func toTCERegion(cluster scheduled_plan_types.Cluster) types.TCERegion {
	return types.TCERegion{
		Zone:            *cluster.Zone,
		Dc:              *cluster.Dc,
		PhysicalCluster: *cluster.PhysicalCluster,
		LogicalCluster:  *cluster.LogicalCluster,
	}
}

func toTCETimeWindow(timeWindow scheduled_plan_types.TimeWindow) types.TimeWindow {
	var endTime *time.Time
	if timeWindow.EndTime != nil {
		endTime = utils.ToPtr(timeWindow.EndTime.UTC())
	}
	return types.TimeWindow{
		StartTime: timeWindow.StartTime.UTC(),
		EndTime:   endTime,
	}
}

func toTCECommitInfo(commitInfo *scheduled_plan_types.CommitInfo) *types.TCECommitInfo {
	if commitInfo == nil {
		return nil
	}

	return &types.TCECommitInfo{
		ResourcePoolName: commitInfo.ResourcePoolName,
	}
}

func toTCETimelineEntries(entries []scheduled_plan_types.MatchTimelineEntry) []types.MatchingOrderTimelineEntry {
	result := make([]types.MatchingOrderTimelineEntry, 0, len(entries))
	for _, e := range entries {
		result = append(result, types.MatchingOrderTimelineEntry{
			NewStatus:        e.NewStatus,
			NewDisplayStatus: e.NewDisplayStatus,
			Event:            e.Event,
			Note:             e.Note,
			CreatedAt:        e.CreatedAt,
		})
	}
	return result
}

func toTCETicketPriority(ticketID int64, tp *gpu_center_client.TicketPriorityResult) *types.TicketPriorityDetail {
	if tp == nil {
		return nil
	}

	return &types.TicketPriorityDetail{
		TicketID:              ticketID,
		Priority:              tp.Priority,
		ResourceGroupPriority: tp.ResourceGroupPriority,
		ResourceGroupWeight:   tp.ResourceGroupWeight,
		GPUUtilPriority:       tp.GPUUtilPriority,
		GPUUtilWeight:         tp.GPUUtilWeight,
		BizPriority:           tp.BizPriority,
		BizWeight:             tp.BizWeight,
		WorkloadPriority:      tp.WorkloadPriority,
		WorkloadWeight:        tp.WorkloadWeight,
		SceneWeight:           tp.SceneWeight,
		PlatformWeight:        tp.PlatformWeight,
	}
}
