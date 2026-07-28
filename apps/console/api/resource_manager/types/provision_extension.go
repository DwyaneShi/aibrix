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

package types

// Only used for extending the custom resource group options.
type ExtensionResourceGroupOptions struct {
	// TCE contains TCE-specific options.
	TCE *TCEGroupOptions `json:"tce,omitempty"`
}

// TCEGroupOptions contains TCE-specific topology options.
type TCEGroupOptions struct {
	// NumaConfig contains NUMA-related configuration.
	NumaConfig *NUMAConfig `json:"numaConfig,omitempty"`

	// ReplicaAffinity constrains how resources within this group should be co-located.
	// Policies: numa (strongest) > host > tor > minipod > bigpod (weakest).
	ReplicaAffinity *AffinityPolicies `json:"replicaAffinity,omitempty"`

	// GroupAffinity constrains how this group should be co-located relative to other groups.
	GroupAffinity *AffinityPolicies `json:"groupAffinity,omitempty"`

	RegionAffinity TCERegionAffinity `json:"regionAffinity"`

	// TopologyConstraint is optional. It can be used to apply specific topology constraints.
	TopologyConstraint *map[string]string `json:"topologyConstraint,omitempty"`
}

// Only used for extending the custom provision result details.
type ExtensionProvisionResultDetails struct {
	// TCE contains TCE-specific provision details (MatchingResult).
	// Set when provider is "tce".
	TCE *TCEProvisionDetail `json:"tce,omitempty"`
}

// MatchingOrderTimelineEntry represents a single state transition in the
// matching order lifecycle, derived from ResourceManager match timeline API.
type MatchingOrderTimelineEntry struct {
	NewStatus        string `json:"new_status"`
	NewDisplayStatus string `json:"new_display_status"`
	Event            string `json:"event"`
	Note             string `json:"note,omitempty"`
	CreatedAt        string `json:"created_at"`
}

// TCEProvisionDetail contains TCE-specific provision result details.
type TCEProvisionDetail struct {
	// MatchId is the matching task ID.
	MatchId string `json:"matchId"`

	MatchOrderUrl string `json:"matchOrderUrl"`

	// GroupResults contains allocation details for each group.
	// Each group can have multiple allocation segments (for elastic scheduling, etc.).
	GroupResults *TCEGroupResults `json:"groupResults,omitempty"`

	// Timeline contains matching order state transition entries, populated during
	// provision polling by calling GetScheduledMatchDetail.
	Timeline []MatchingOrderTimelineEntry `json:"timeline,omitempty"`

	TicketPriority *TicketPriorityDetail `json:"ticketPriority,omitempty"`
}

// TCEGroupResults is a list of group allocation results.
type TCEGroupResults = []TCEGroupResult

// TCEGroupResult contains allocation details for a single group.
type TCEGroupResult struct {
	// GroupRole is the role of this group (if specified in request).
	GroupRole *string `json:"groupRole,omitempty"`

	// AllocationSegments contains allocation details for each time segment.
	// Supports elastic scheduling, checkpoint recovery, etc.
	AllocationSegments []TCEAllocationSegment `json:"allocationSegments"`
}

// TCEAllocationSegment describes a single allocation segment.
type TCEAllocationSegment struct {
	// Id is the allocation segment ID.
	Id string `json:"id"`

	// Allocated indicates whether this segment was successfully allocated.
	Allocated bool `json:"allocated"`

	// Region contains the region information (zone, dc, physicalCluster, etc.).
	Region TCERegion `json:"region"`

	// AcceleratorType is the allocated accelerator type (e.g., "A100", "H100").
	AcceleratorType string `json:"acceleratorType"`

	// AcceleratorCategory is the accelerator category (gpu/xpu/npu).
	AcceleratorCategory string `json:"acceleratorCategory"`

	// Count is the number of allocated accelerators.
	Count *int `json:"count,omitempty"`

	// Replicas is the number of replicas allocated.
	Replicas *int `json:"replicas,omitempty"`

	// NodeIds is the list of allocated node IDs.
	NodeIds []string `json:"nodeIds"`

	// AcceleratorIds is the list of allocated accelerator IDs.
	AcceleratorIds []string `json:"acceleratorIds"`

	// TimeWindow is the allocation time window.
	TimeWindow TimeWindow `json:"timeWindow"` // TimeWindow type

	// Preemptible indicates whether this allocation can be preempted.
	Preemptible *bool `json:"preemptible,omitempty"`

	// CommitInfo contains commit-related information.
	CommitInfo *TCECommitInfo `json:"commitInfo,omitempty"`
}

// TCECommitInfo contains commit-related information.
type TCECommitInfo struct {
	// ResourcePoolName is the resource pool name.
	ResourcePoolName *string `json:"resourcePoolName,omitempty"`
}

// TicketPriorityDetail contains fusion priority calculation details from GPU Center.
type TicketPriorityDetail struct {
	TicketID              int64   `json:"ticketId"`
	Priority              int64   `json:"priority"`
	ResourceGroupPriority int64   `json:"resourceGroupPriority"`
	ResourceGroupWeight   float64 `json:"resourceGroupWeight"`
	GPUUtilPriority       int64   `json:"gpuUtilPriority"`
	GPUUtilWeight         float64 `json:"gpuUtilWeight"`
	BizPriority           int64   `json:"bizPriority"`
	BizWeight             float64 `json:"bizWeight"`
	WorkloadPriority      int64   `json:"workloadPriority"`
	WorkloadWeight        float64 `json:"workloadWeight"`
	SceneWeight           float64 `json:"sceneWeight"`
	PlatformWeight        float64 `json:"platformWeight"`
}
