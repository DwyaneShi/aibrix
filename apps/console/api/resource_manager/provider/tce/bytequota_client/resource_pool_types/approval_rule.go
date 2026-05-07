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

type GroupApprovalRule struct {
	AutoApprovalRule       *GroupAutoApprovalRule       `json:"auto_approval_rule"         bson:"auto_approval_rule"`
	ArtificialApprovalRule *GroupArtificialApprovalRule `json:"artificial_approval_rule"   bson:"artificial_approval_rule"`
}

type GroupAutoApprovalRule struct {
	IsOpen           bool                `json:"is_open"                     bson:"is_open"`
	ApprovalRuleType string              `json:"approval_rule_type"          bson:"approval_rule_type"`
	NotOverCPUBudget bool                `json:"not_over_cpu_budget"         bson:"not_over_cpu_budget"`
	GeneralRule      *GroupGeneralRule   `json:"general_rule"                bson:"general_rule"`
	CustomizeRule    *GroupCustomizeRule `json:"customize_rule"              bson:"customize_rule"`
}

type GroupGeneralRule struct {
	ApplyLimitPerPersonPerDay map[string]map[string]int64 `json:"apply_limit_per_person_per_day"     bson:"apply_limit_per_person_per_day"`
}

type GroupCustomizeRule struct {
	WebhookURL string `json:"webhook_url" bson:"webhook_url"`
}

type GroupArtificialApprovalRule struct {
	IsOpen        bool     `json:"is_open"             bson:"is_open"`
	Approver      []string `json:"approver"            bson:"approver"`
	DutyQueues    []string `json:"duty_queues"         bson:"duty_queues"`
	DutyApprovers []string `json:"duty_approvers"      bson:"duty_approvers"`
}

type DomainApprovalRule struct {
	AutoApprovalRule       *DomainAutoApprovalRule       `json:"auto_approval_rule"`
	ArtificialApprovalRule *DomainArtificialApprovalRule `json:"artificial_approval_rule"`
}

type DomainAutoApprovalRule struct {
	IsOpen              bool                   `json:"is_open"`
	DomainGeneralRule   *DomainGeneralRule     `json:"domain_general_rule"`
	DomainCustomizeRule []*DomainCustomizeRule `json:"domain_customize_rule"`
}

type DomainArtificialApprovalRule struct {
	IsOpen        bool     `json:"is_open"`
	Approver      []string `json:"approver"`
	DutyQueues    []string `json:"duty_queues"`
	DutyApprovers []string `json:"duty_approvers"`
}

type DomainFreeApprovalConfig struct {
	EnableFreeApprovalResourceValue      bool                          `json:"enable_free_approval_resource_value"`
	EnableFreeApprovalResourcePercentage bool                          `json:"enable_free_approval_resource_percentage"`
	FreeApprovalResourceValue            ResourceItem                  `json:"free_approval_resource_value"`
	FreeApprovalResourcePercentage       map[string]map[string]float64 `json:"free_approval_resource_percentage"`
	FreeApprovalResourcePercentageValue  ResourceItem                  `json:"free_approval_resource_percentage_value"`
}

type DomainGeneralRule struct {
	DomainFreeApprovalConfig
}

type DomainCustomizeRule struct {
	BusinessLine string `json:"business_line"`
	DomainFreeApprovalConfig
}
