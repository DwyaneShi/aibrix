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

type Budget struct {
	ID            uint64       `json:"id"`
	Name          string       `json:"name"`
	NodeIDs       []uint64     `json:"node_ids"`
	Product       string       `json:"product"`
	EffectiveTime []string     `json:"effective_time"`
	Budget        ResourceItem `json:"budget"`
	Admins        []string     `json:"admins"`
	BillCount     ResourceItem `json:"bill_count"`
}

type BudgetInfo struct {
	Budgets []*Budget   `json:"budgets"`
	Quota   QuotaSupply `json:"quota"`
	Admins  []string    `json:"admins"`
}

type SharingScopeMeta struct {
	ID       uint64 `json:"id"`
	Name     string `json:"name"`
	I18NName string `json:"i18n_name,omitempty"`
	// Level is the same with node level on service tree.
	// The min scope Level is 1. 0 is invalid
	Level uint `json:"level"`
}

// ResourceGroup for platform to abstract resource pools group
type ResourceGroupMeta struct {
	ID                string                             `json:"id"                         bson:"id"`
	Name              string                             `json:"name"                       bson:"name"`
	I18NName          string                             `json:"i18n_name,omitempty"        bson:"i18n_name"`
	Platform          string                             `json:"platform"                   bson:"platform"`
	SharingScopeIDs   []uint64                           `json:"sharing_scope_ids"          bson:"sharing_scope_ids"`
	Admins            []string                           `json:"admins"                     bson:"admins"`
	ServiceAccounts   []string                           `json:"service_accounts"           bson:"service_accounts"`
	DutyQueues        []string                           `json:"duty_queues"                bson:"duty_queues"`
	DutyAdmins        []string                           `json:"duty_admins"                bson:"duty_admins"`
	ApprovalRule      map[string]*GroupApprovalRule      `json:"approval_rule"              bson:"approval_rule"`
	ConsumeReviewInfo map[string]*GroupConsumeReviewInfo `json:"consume_review_info"        bson:"consume_review_info"`
}

type GroupConsumeReviewInfo struct {
	ConsumeReview      bool                `json:"consume_review"             bson:"consume_review"`
	ConsumeReviewType  string              `json:"consume_review_type"        bson:"consume_review_type"`
	FreeApprovalConfig *FreeApprovalConfig `json:"free_approval_config"       bson:"free_approval_config"`
}
