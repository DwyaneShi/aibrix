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

package resource_pool_types

import (
	"strconv"
	"strings"
	"time"
)

type ListResourcePoolsRequest struct {
	Name                      *string   `form:"name"`
	Psm                       *string   `form:"psm"`
	ByteTreeID                *string   `form:"bytetree_id"`
	Summary                   *bool     `form:"summary"`
	Detail                    *bool     `form:"detail"`
	Aggregated                *bool     `form:"aggregated"`
	ConvertUnit               *bool     `form:"convert_unit"`
	ExcludeZero               *bool     `form:"exclude_zero"`
	WithNotExistAndFreezed    *bool     `form:"with_not_exist_and_freezed"`
	OrderBy                   *string   `form:"order_by"`
	LimitStr                  *string   `form:"limit"`
	SupportGpu                *string   `form:"support_gpu"`
	SupportHabana             *string   `form:"support_habana"`
	SupportCodec              *string   `form:"support_codec"`
	SupportXpu                *string   `form:"support_xpu"`
	SupportNpu                *string   `form:"support_npu"`
	SupportNic                *string   `form:"support_nic"`
	AllowNormalResource       *string   `form:"allow_normal_resource"`
	AllowOversoldResource     *string   `form:"allow_oversold_resource"`
	AllowExclusiveResource    *string   `form:"allow_exclusive_resource"`
	UseSocket                 *string   `form:"use_socket"`
	IsIntegrated              *string   `form:"is_integrated"`
	IsReservation             *bool     `form:"is_reservation"`
	DisableConfigPackages     *bool     `form:"disable_config_packages"`
	ScopeIDArr                *[]string `form:"scope_id"`
	AZ                        *string   `form:"az"`
	Platform                  *string   `form:"platform"`
	Admin                     *string   `form:"admin"`
	ResourceGroupID           *string   `form:"resource_group_id"`
	ConsumeReviewOnly         *bool     `form:"consume_review_only"`
	PoolAlertOnly             *bool     `form:"pool_alert_only"`
	ExcludeFreezed            *bool     `form:"exclude_freezed"`
	IncludeDeleted            *bool     `form:"include_deleted"`
	ResourceGroupIDs          *[]string `form:"resource_group_ids"`
	ClusterNames              *string   `form:"cluster_names"`
	Synced                    *bool     `form:"synced"`
	PhysicalClusters          *[]string `form:"physical_clusters"`
	SupplyDomains             *[]string `form:"supply_domains"`
	OnlyFederatedGPUPartition *bool     `form:"only_federated_gpu_partition"`
	WithoutScheduled          *bool     `form:"without_scheduled"`
	WithoutOndemand           *bool     `form:"without_ondemand"`
}

func (r *ListResourcePoolsRequest) GetParams() map[string]string {
	params := map[string]string{}
	if r.Name != nil {
		params["name"] = *r.Name
	}
	if r.Psm != nil {
		params["psm"] = *r.Psm
	}
	if r.ByteTreeID != nil {
		params["bytetree_id"] = *r.ByteTreeID
	}
	if r.Summary != nil {
		params["summary"] = strconv.FormatBool(*r.Summary)
	}
	if r.Detail != nil {
		params["detail"] = strconv.FormatBool(*r.Detail)
	}
	if r.Aggregated != nil {
		params["aggregated"] = strconv.FormatBool(*r.Aggregated)
	}
	if r.ConvertUnit != nil {
		params["convert_unit"] = strconv.FormatBool(*r.ConvertUnit)
	}
	if r.ExcludeZero != nil {
		params["exclude_zero"] = strconv.FormatBool(*r.ExcludeZero)
	}
	if r.WithNotExistAndFreezed != nil {
		params["with_not_exist_and_freezed"] = strconv.FormatBool(*r.WithNotExistAndFreezed)
	}
	if r.OrderBy != nil {
		params["order_by"] = *r.OrderBy
	}
	if r.LimitStr != nil {
		params["limit"] = *r.LimitStr
	}
	if r.SupportGpu != nil {
		params["support_gpu"] = *r.SupportGpu
	}
	if r.SupportHabana != nil {
		params["support_habana"] = *r.SupportHabana
	}
	if r.SupportCodec != nil {
		params["support_codec"] = *r.SupportCodec
	}
	if r.SupportXpu != nil {
		params["support_xpu"] = *r.SupportXpu
	}
	if r.SupportNpu != nil {
		params["support_npu"] = *r.SupportNpu
	}
	if r.SupportNic != nil {
		params["support_nic"] = *r.SupportNic
	}
	if r.AllowNormalResource != nil {
		params["allow_normal_resource"] = *r.AllowNormalResource
	}
	if r.AllowOversoldResource != nil {
		params["allow_oversold_resource"] = *r.AllowOversoldResource
	}
	if r.AllowExclusiveResource != nil {
		params["allow_exclusive_resource"] = *r.AllowExclusiveResource
	}
	if r.UseSocket != nil {
		params["use_socket"] = *r.UseSocket
	}
	if r.IsIntegrated != nil {
		params["is_integrated"] = *r.IsIntegrated
	}
	if r.IsReservation != nil {
		params["is_reservation"] = strconv.FormatBool(*r.IsReservation)
	}
	if r.DisableConfigPackages != nil {
		params["disable_config_packages"] = strconv.FormatBool(*r.DisableConfigPackages)
	}
	if r.ScopeIDArr != nil && len(*r.ScopeIDArr) > 0 {
		params["scope_id"] = strings.Join(*r.ScopeIDArr, ",")
	}
	if r.AZ != nil {
		params["az"] = *r.AZ
	}
	if r.Platform != nil {
		params["platform"] = *r.Platform
	}
	if r.Admin != nil {
		params["admin"] = *r.Admin
	}
	if r.ResourceGroupID != nil {
		params["resource_group_id"] = *r.ResourceGroupID
	}
	if r.ConsumeReviewOnly != nil {
		params["consume_review_only"] = strconv.FormatBool(*r.ConsumeReviewOnly)
	}
	if r.PoolAlertOnly != nil {
		params["pool_alert_only"] = strconv.FormatBool(*r.PoolAlertOnly)
	}
	if r.ExcludeFreezed != nil {
		params["exclude_freezed"] = strconv.FormatBool(*r.ExcludeFreezed)
	}
	if r.IncludeDeleted != nil {
		params["include_deleted"] = strconv.FormatBool(*r.IncludeDeleted)
	}
	if r.ResourceGroupIDs != nil && len(*r.ResourceGroupIDs) > 0 {
		params["resource_group_ids"] = strings.Join(*r.ResourceGroupIDs, ",")
	}
	if r.ClusterNames != nil {
		params["cluster_names"] = *r.ClusterNames
	}
	if r.Synced != nil {
		params["synced"] = strconv.FormatBool(*r.Synced)
	}
	if r.PhysicalClusters != nil && len(*r.PhysicalClusters) > 0 {
		params["physical_clusters"] = strings.Join(*r.PhysicalClusters, ",")
	}
	if r.SupplyDomains != nil && len(*r.SupplyDomains) > 0 {
		params["supply_domains"] = strings.Join(*r.SupplyDomains, ",")
	}
	if r.OnlyFederatedGPUPartition != nil {
		params["only_federated_gpu_partition"] = strconv.FormatBool(*r.OnlyFederatedGPUPartition)
	}
	if r.WithoutScheduled != nil {
		params["without_scheduled"] = strconv.FormatBool(*r.WithoutScheduled)
	}
	if r.WithoutOndemand != nil {
		params["without_ondemand"] = strconv.FormatBool(*r.WithoutOndemand)
	}

	return params
}

type FreeApprovalConfig struct {
	GeneralFreeApprovalConfig   GeneralFreeApprovalConfig   `json:"general_free_approval_config"    bson:"general_free_approval_config"`
	CustomizeFreeApprovalConfig CustomizeFreeApprovalConfig `json:"customize_free_approval_config"  bson:"customize_free_approval_config"`
}

type GeneralFreeApprovalConfig struct {
	EnableFreeApprovalByCPUPercentage bool `json:"enable_free_approval_by_cpu_percentage"    bson:"enable_free_approval_by_cpu_percentage"`
	EnableFreeApprovalByResource      bool `json:"enable_free_approval_by_resource"          bson:"enable_free_approval_by_resource"`

	FreeApprovalCPUPercentage float64      `json:"free_approval_cpu_percentage"              bson:"free_approval_cpu_percentage"`
	UseRealTimeCPUPercentage  bool         `json:"use_real_time_cpu_percentage"              bson:"use_real_time_cpu_percentage"`
	UseHistoryCPUPercentage   bool         `json:"use_history_cpu_percentage"                bson:"use_history_cpu_percentage"`
	FreeApprovalResource      ResourceItem `json:"free_approval_resource"                    bson:"free_approval_resource"`
}

type CustomizeFreeApprovalConfig struct {
	WebhookURL string `json:"webhook_url"      bson:"webhook_url"`
}

// ResourcePoolMeta to identify resource pool
type ResourcePoolMeta struct {
	Name               string              `json:"name"                            bson:"name"`                 // resource pool name
	Namespace          string              `json:"namespace"                       bson:"namespace"`            // resource pool namespace
	Alias              string              `json:"alias"                           bson:"alias"`                // resource pool alias
	Platform           string              `json:"platform"                        bson:"platform"`             // platform
	ResourceGroupID    string              `json:"resource_group_id"               bson:"resource_group_id"`    // resource group id
	Labels             map[string]string   `json:"labels"                          bson:"labels"`               // Labels contains the required fields
	Annotations        map[string]string   `json:"annotations"                     bson:"annotations"`          // Annotations contains the optional fields
	ConsumeReview      bool                `json:"consume_review"                  bson:"consume_review"`       // consume review
	ConsumeReviewType  string              `json:"consume_review_type"             bson:"consume_review_type"`  // consume review type
	FreeApprovalConfig *FreeApprovalConfig `json:"free_approval_config"            bson:"free_approval_config"` // free consume review config
	Freezed            bool                `json:"freezed"                         bson:"freezed"`              // is resource pool freezed
	Admins             []string            `json:"admins"                          bson:"admins"`               // BDEE/USTS employees, admin names, the admins of resource pool
	ServiceAccounts    []string            `json:"service_accounts"                bson:"namservice_accounts"`  // sservice_accountsrvice accounts
	DutyQueues         []string            `json:"duty_queues"                     bson:"duty_queues"`          // duty queues
	DutyAdmins         []string            `json:"duty_admins"                     bson:"duty_admins"`          // BDEE/USTS duty_admins, duty admin names, the duty admins of resource pool
	LockingUpdatedAt   time.Time           `json:"locking_updated_at"              bson:"locking_updated_at"`   // locking updated time
	ApproveResource    ResourceItem        `json:"approve_resource"                bson:"approve_resource"`     // approve resource, contains cpu, memory, socket and gpu
	Alert              bool                `json:"alert"                           bson:"alert"`                // need alert or not
	AlertResource      ResourceItem        `json:"alert_resource"                  bson:"alert_resource"`       // alert resource, contains cpu, memory, socket and gpu
	Qos                string              `json:"qos"                             bson:"qos"`                  // qos, type of resource
	SyncStatus         string              `json:"sync_status"                     bson:"sync_status"`          // sync status means the status of syncing to k8s
	BytetreeID         uint64              `json:"bytetree_id" bson:"bytetree_id"`                              // bytetree id for resource pool

	// 离线场景：
	//  所有情况：shared, resource_type=cpu/mem
	// 在线场景：
	//  Socket only集群：dedicated, resource_type=socket
	// 	Socket support集群：dedicated, resource_type=cpu/mem
	// 	小容器集群：shared, resource_type=cpu/mem
	qos string
}

// CommonQuota to hold resource items for common quota requirment
type CommonQuota struct {
	Capacity    ResourceItem `json:"capacity"`
	IncCapacity ResourceItem `json:"inc_capacity"`

	// used in uce
	BufferQuota     ResourceItem `json:"buffer_quota"`
	OndemandV2Quota ResourceItem `json:"ondemand_v2_quota"`
	SpotQuota       ResourceItem `json:"spot_quota"`

	LimitCapacity    ResourceItem           `json:"limit_capacity"` // derive field
	RealtimeCapacity ResourceItem           `json:"realtime_capacity"`
	Available        ResourceItem           `json:"available"`       // derive field
	IncAvailable     ResourceItem           `json:"inc_available"`   // derive field
	LimitAvailable   ResourceItem           `json:"limit_available"` // derive field
	Usage            ResourceItem           `json:"usage"`
	Locked           ResourceItem           `json:"locked"`
	ActualUsage      ResourceItem           `json:"actual_usage"`
	Reservation      ResourceItem           `json:"reservation"` // 总量
	Reservations     []*ResourceReservation `json:"reservations"`
	QuotaOnDemand    QuotaSpecOnDemand      `json:"quota_spec_ondemand"`
	ScheduledQuota   *ScheduledQuotaSpec    `json:"scheduled_quota"`

	DedicatedQuota QuotaSaleMode `json:"dedicated_quota"`
	SharedQuota    QuotaSaleMode `json:"shared_quota"`
	ReclaimedQuota QuotaSaleMode `json:"reclaimed_quota"`
}

type QuotaSaleMode struct {
	Reserved       QuotaSpec          `json:"reserved"`
	Spot           QuotaSpec          `json:"spot"`
	OnDemand       QuotaSpecOnDemand  `json:"ondemand"`
	ScheduledQuota TimeScheduledQuota `json:"scheduled"`
	Scheduled      ScheduledQuotaSpec `json:"-"`
	Buffer         QuotaSpec          `json:"buffer"`
	OndemandV2     QuotaSpec          `json:"ondemand_v2"`
}

type TimeScheduledQuota struct {
	TimeSpecifiedCapacity  TimeResourceItem `json:"time_specified_capacity"`
	TimeSpecifiedAvaliable TimeResourceItem `json:"time_specified_available"`
	TimeSpecifiedUsage     TimeResourceItem `json:"time_specified_usage"`
}

type QuotaSpec struct {
	Capacity               ResourceItem     `json:"capacity"`
	Usage                  ResourceItem     `json:"usage"`
	Available              ResourceItem     `json:"available"`
	Reservation            ResourceItem     `json:"reservation"` // 总量
	TimeSpecifiedAllocated TimeResourceItem `json:"time_specified_allocated"`
}

// PackageQuota store quota info for tce socket package
type PackageQuota struct {
	Available          map[string]int64            `json:"available"`
	IncAvailable       map[string]int64            `json:"inc_available"`
	LimitAvailable     map[string]int64            `json:"limit_available"`
	Usage              map[string]int64            `json:"usage"`
	OnDemandUsage      map[string]int64            `json:"ondemand_usage"`
	OnDemandAvailable  map[string]int64            `json:"ondemand_available"`
	Reservation        map[string]int64            `json:"reservation"`
	PsmToReservation   map[string]map[string]int64 `json:"psm_to_reservation"`
	SceneToReservation map[string]map[string]int64 `json:"scene_to_reservation"`
}

type ResourceGroup struct {
	RecordMeta
	ResourceGroupMeta
	SharingScopes []*SharingScopeMeta `json:"sharing_scopes"`
	ResourcePools []*ResourcePoolMeta `json:"resource_pools,omitempty"`
}

type groupResp struct {
	*ResourceGroup
	BudgetInfo *BudgetInfo `json:"budget_info,omitempty"`
	Collection bool        `json:"collection"`
}

type PoolFeature struct {
	AllowNormalResource    bool `json:"allow_normal_resource"`
	AllowOversoldResource  bool `json:"allow_oversold_resource"`
	AllowExclusiveResource bool `json:"allow_exclusive_resource"`
}

type PoolStatus struct {
	Exist bool `json:"exist"`
}

type ResourcePoolResp struct {
	*ResourcePoolMeta
	*RecordMeta
	Quota            *CommonQuota         `json:"quota,omitempty"`
	Package          *PackageQuota        `json:"package,omitempty"`
	ExclusivePackage *PackageQuota        `json:"exclusive_package"`
	PackageOnDemand  *PackageSpecOnDemand `json:"package_ondemand"`
	DedicatedPackage *PackageSaleMode     `json:"dedicated_package"`
	ResourceGroup    *groupResp           `json:"resource_group,omitempty"`
	SupplyDomain     *SupplyDomainMeta    `json:"supply_domain,omitempty"`
	Features         *PoolFeature         `json:"features,omitempty"`
	Status           *PoolStatus          `json:"status,omitempty"`
}

type GetResourceGroupRequest struct {
	Platform                  string   `json:"platform"`
	SupplyDomains             []string `json:"supply_domains"`
	OnlyFederatedGPUPartition bool     `json:"only_federated_gpu_partition"`
	QueueName                 string   `json:"queue_name"`
	Type                      string   `json:"type"`
	Env                       string   `json:"env"`
}

type GetResourceGroupResponse struct {
	Code    int            `json:"error_code"`
	Message string         `json:"message"`
	Data    *ResourceGroup `json:"data"`
}
