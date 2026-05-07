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

type DomainFeatures struct {
	UseSocket                bool `json:"use_socket"`
	IsIntegrated             bool `json:"is_integrated"`
	SupportGpu               bool `json:"support_gpu"`
	SupportHabana            bool `json:"support_habana"`
	SupportCodec             bool `json:"support_codec"`
	SupportXpu               bool `json:"support_xpu"`
	SupportNpu               bool `json:"support_npu"`
	SupportNic               bool `json:"support_nic"`
	SupportNbw               bool `json:"support_nbw"`
	SupportNormalResource    bool `json:"support_normal_resource"`
	SupportOversoldResource  bool `json:"support_oversold_resource"`
	SupportExclusiveResource bool `json:"support_exclusive_resource"`
	SupportFlexSocket        bool `json:"support_flex_socket"`
}

type UrgentResource struct {
	ResourcePoolName      string                 `json:"resource_pool_name"`
	ResourceGroupName     string                 `json:"resource_group_name"`
	ResourceGroupID       string                 `json:"resource_group_id"`
	SharingScopeName      string                 `json:"sharing_scope_name"`
	FreeApprovalResources *FreeApprovalResources `json:"free_approval_resources"`
}

type FreeApprovalResources struct {
	FreeApprovalResourcesCapacity  ResourceItem `json:"free_approval_resources_capacity"`
	FreeApprovalResourcesAvailable ResourceItem `json:"free_approval_resources_available"`
	FreeApprovalResourcesUsage     ResourceItem `json:"free_approval_resources_usage"`
}

type LowAvailabilityApproval struct {
	IsOpen            bool         `json:"is_open"`
	NeedApproveConfig ResourceItem `json:"need_approve_config"`
}

type SupplyDomainMeta struct {
	Name                    string                         `bson:"name" json:"name"`
	Platform                string                         `bson:"platform" json:"platform"`
	Labels                  map[string]string              `bson:"labels" json:"labels"`
	Annotations             map[string]string              `bson:"annotations" json:"annotations"`
	Buffer                  ResourceItem                   `bson:"buffer" json:"buffer"`
	Freezed                 bool                           `bson:"freezed" json:"freezed"`
	Features                DomainFeatures                 `bson:"features" json:"features"`
	Admins                  []string                       `bson:"admins" json:"admins"`
	DutyQueues              []string                       `bson:"duty_queues" json:"duty_queues"`
	DutyAdmins              []string                       `bson:"duty_admins" json:"duty_admins"`
	ReserveRatio            map[string]map[string]float64  `bson:"reserve_ratio" json:"reserve_ratio"`
	ApprovalRule            map[string]*DomainApprovalRule `bson:"approval_rule" json:"approval_rule"`
	EnableConfigPackages    bool                           `bson:"enable_config_packages" json:"enable_config_packages"`
	ConfigPackages          []string                       `bson:"config_packages" json:"config_packages"`
	UrgentResources         []*UrgentResource              `bson:"urgent_resources" json:"urgent_resources"`
	ServiceAccounts         []string                       `bson:"service_accounts" json:"service_accounts"`
	LowAvailabilityApproval *LowAvailabilityApproval       `bson:"low_availability_approval" json:"low_availability_approval"`
}
