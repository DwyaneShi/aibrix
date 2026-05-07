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

package scheduled_plan_types

import "github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/utils"

type ScheduledPlanTicketDisplayStatus string

// 工单展示状态枚举说明：
// - TICKET_SUBMITTED：创建后，且目前无任何撮合记录
// - MATCH_MATCHING：目前有撮合记录，但无任何结果
// - MATCH_SUCCEEDED：撮合成功
// - MATCH_FAILED：撮合失败
// - MATCH_CANCELLED：产生最终撮合结果前取消
// - RESOURCE_COMMITTED_FAILED：撮合成功后，资源交付失败
// - RESOURCE_PENDING_EFFECTIVE：资源已确认，目前未到开始时间
// - RESOURCE_EARLY_CANCELLED：资源已确认后取消，取消时间未到开始时间
// - RESOURCE_IN_EFFECT：资源已确认，目前已到开始时间
// - RESOURCE_EARLY_RELEASED：资源已确认，取消时间在开始时间后
// - RESOURCE_EXPIRED：资源已确认，目前已过结束时间

const (
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_TICKET_SUBMITTED           ScheduledPlanTicketDisplayStatus = "ticket_submitted"           // 创建后，且目前无任何撮合记录
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_MATCHING             ScheduledPlanTicketDisplayStatus = "match_matching"             // 目前有撮合记录，但无任何结果
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_SUCCEEDED            ScheduledPlanTicketDisplayStatus = "match_succeeded"            // 撮合成功
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_FAILED               ScheduledPlanTicketDisplayStatus = "match_failed"               // 撮合失败
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_CANCELLED            ScheduledPlanTicketDisplayStatus = "match_cancelled"            // 产生最终撮合结果前取消
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_COMMITTED_FAILED  ScheduledPlanTicketDisplayStatus = "resource_committed_failed"  // 撮合成功后，资源交付失败
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_PENDING_EFFECTIVE ScheduledPlanTicketDisplayStatus = "resource_pending_effective" // 资源已确认，目前未到开始时间
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EARLY_CANCELLED   ScheduledPlanTicketDisplayStatus = "resource_early_cancelled"   // 资源已确认后取消，取消时间未到开始时间
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_IN_EFFECT         ScheduledPlanTicketDisplayStatus = "resource_in_effect"         // 资源已确认，目前已到开始时间
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EARLY_RELEASED    ScheduledPlanTicketDisplayStatus = "resource_early_released"    // 资源已确认，取消时间在开始时间后
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EXPIRED           ScheduledPlanTicketDisplayStatus = "resource_expired"           // 资源已确认，目前已过结束时间
)

var CreatingStageScheduledPlanTicketDisplayStatuses = []ScheduledPlanTicketDisplayStatus{
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_TICKET_SUBMITTED,
}

var MatchingStageScheduledPlanTicketDisplayStatuses = []ScheduledPlanTicketDisplayStatus{
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_MATCHING,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_SUCCEEDED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_FAILED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_CANCELLED,
}

var CommittingStageScheduledPlanTicketDisplayStatuses = []ScheduledPlanTicketDisplayStatus{
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_COMMITTED_FAILED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_PENDING_EFFECTIVE,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EARLY_CANCELLED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_IN_EFFECT,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EARLY_RELEASED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EXPIRED,
}

var AllScheduledPlanTicketDisplayStatuses = []ScheduledPlanTicketDisplayStatus{
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_TICKET_SUBMITTED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_MATCHING,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_SUCCEEDED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_FAILED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_MATCH_CANCELLED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_COMMITTED_FAILED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_PENDING_EFFECTIVE,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EARLY_CANCELLED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_IN_EFFECT,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EARLY_RELEASED,
	SCHEDULED_PLAN_TICKET_DISPLAY_STATUS_RESOURCE_EXPIRED,
}

func (s ScheduledPlanTicketDisplayStatus) IsCreatingStage() bool {
	return utils.SliceContain(CreatingStageScheduledPlanTicketDisplayStatuses, s)
}

func (s ScheduledPlanTicketDisplayStatus) IsMatchingStage() bool {
	return utils.SliceContain(MatchingStageScheduledPlanTicketDisplayStatuses, s)
}

func (s ScheduledPlanTicketDisplayStatus) IsCommittingStage() bool {
	return utils.SliceContain(CommittingStageScheduledPlanTicketDisplayStatuses, s)
}

// turn []ScheduledPlanTicketDisplayStatus to []string
func ScheduledPlanTicketDisplayStatusesToStrings(statuses []ScheduledPlanTicketDisplayStatus) []string {
	var statusStrs []string
	for _, status := range statuses {
		statusStrs = append(statusStrs, string(status))
	}
	return statusStrs
}
