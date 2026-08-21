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

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/utils"
)

type MatchingDetailResponse struct {
	LastDecisionTime int64                            `json:"lastDecisionTime"`
	DecisionDeadline int64                            `json:"decisionDeadline"`
	Intent           *MatchingIntent                  `json:"intent"`
	State            MatchingIntentStatus             `json:"state"`
	LastState        MatchingIntentStatus             `json:"last_state"`
	Result           *MatchingResult                  `json:"result"`
	DebugInfo        map[string]any                   `json:"debugInfo"`
	ID               int64                            `json:"id"`
	Name             string                           `json:"name"`
	Description      string                           `json:"description"`
	Priority         int64                            `json:"priority"`
	Operator         string                           `json:"operator"`
	CanceledOperator string                           `json:"canceled_operator"`
	CreatedAt        time.Time                        `json:"created_at"`
	UpdatedAt        time.Time                        `json:"updated_at"`
	LastScheduledAt  time.Time                        `json:"last_scheduled_at"`
	BookedAt         *time.Time                       `json:"booked_at"`
	SucceedAt        *time.Time                       `json:"succeed_at"`
	CancelledAt      *time.Time                       `json:"cancelled_at"`
	FailedAt         *time.Time                       `json:"failed_at"`
	ResourcePoolName string                           `json:"resource_pool_name"`
	DisplayStatus    ScheduledPlanTicketDisplayStatus `json:"display_status"`
	CanceledReason   string                           `json:"canceled_reason"`
}

type ListMatchQuery struct {
	PN   int64  `form:"pn"`
	RN   int64  `form:"rn"`
	Name string `form:"name"`

	// 使用人
	Operator string `form:"operator"`

	// 订单状态
	Status string `form:"status"`

	StatusInt *int64

	// 订单显示状态
	DisplayStatus   string   `form:"display_status"`
	DisplayStatuses []string `form:"display_statuses"` // 多选，用逗号分割

	// 资源组
	ResourceGroupID *int64 `form:"resource_group_id"`

	// 业务线（deprecated）
	BusinessLineID   *int64 `form:"business_line_id"`
	BusinessLineName string `form:"business_line_name"`

	// babi unit
	BabiUnits    []string `form:"babi_units"`
	SubBabiUnits []string `form:"sub_babi_units"`

	// 资源使用时间
	StartTime *time.Time `form:"start_time"`
	EndTime   *time.Time `form:"end_time"`

	// 下单平台
	Platform  *string  `form:"platform"`
	Platforms []string `form:"platforms"` // 多选，用逗号分割

	// 以下暂未实现

	// 下单时间
	SubmitStartTime *time.Time `form:"submit_start_time"`
	SubmitEndTime   *time.Time `form:"submit_end_time"`

	// 融合优先级 MinPriority <= x <= MaxPriority
	MinPriority *int64 `form:"min_priority"`
	MaxPriority *int64 `form:"max_priority"`

	// 卡型
	AcceleratorType  *string  `form:"accelerator_type"`
	AcceleratorTypes []string `form:"accelerator_types"` // 多选，用逗号分割

	// 排序
	OrderByCreatedAt int `form:"sort_by_created_at"`

	// 排序
	OrderByPriority int `form:"sort_by_priority"`
}

func (q *ListMatchQuery) GetParams() map[string]string {
	result := make(map[string]string)

	// 基础字段
	if q.PN != 0 {
		result["pn"] = strconv.FormatInt(q.PN, 10)
	}
	if q.RN != 0 {
		result["rn"] = strconv.FormatInt(q.RN, 10)
	}
	if q.Name != "" {
		result["name"] = q.Name
	}
	if q.Operator != "" {
		result["operator"] = q.Operator
	}
	if q.Status != "" {
		result["status"] = q.Status
	}

	// 指针字段
	if q.StatusInt != nil {
		result["status_int"] = strconv.FormatInt(*q.StatusInt, 10)
	}

	// 字符串字段
	if q.DisplayStatus != "" {
		result["display_status"] = q.DisplayStatus
	}

	// 切片字段（转换为逗号分隔）
	if len(q.DisplayStatuses) > 0 {
		result["display_statuses"] = strings.Join(q.DisplayStatuses, ",")
	}

	if q.ResourceGroupID != nil {
		result["resource_group_id"] = strconv.FormatInt(*q.ResourceGroupID, 10)
	}

	if q.BusinessLineID != nil {
		result["business_line_id"] = strconv.FormatInt(*q.BusinessLineID, 10)
	}

	if q.BusinessLineName != "" {
		result["business_line_name"] = q.BusinessLineName
	}

	// 切片字段
	if len(q.BabiUnits) > 0 {
		result["babi_units"] = strings.Join(q.BabiUnits, ",")
	}

	if len(q.SubBabiUnits) > 0 {
		result["sub_babi_units"] = strings.Join(q.SubBabiUnits, ",")
	}

	// 时间字段
	if q.StartTime != nil && !q.StartTime.IsZero() {
		result["start_time"] = q.StartTime.Format(time.RFC3339)
	}

	if q.EndTime != nil && !q.EndTime.IsZero() {
		result["end_time"] = q.EndTime.Format(time.RFC3339)
	}

	if q.Platform != nil {
		result["platform"] = *q.Platform
	}

	if len(q.Platforms) > 0 {
		result["platforms"] = strings.Join(q.Platforms, ",")
	}

	if q.SubmitStartTime != nil && !q.SubmitStartTime.IsZero() {
		result["submit_start_time"] = q.SubmitStartTime.Format(time.RFC3339)
	}

	if q.SubmitEndTime != nil && !q.SubmitEndTime.IsZero() {
		result["submit_end_time"] = q.SubmitEndTime.Format(time.RFC3339)
	}

	if q.MinPriority != nil {
		result["min_priority"] = strconv.FormatInt(*q.MinPriority, 10)
	}

	if q.MaxPriority != nil {
		result["max_priority"] = strconv.FormatInt(*q.MaxPriority, 10)
	}

	if q.AcceleratorType != nil {
		result["accelerator_type"] = *q.AcceleratorType
	}

	if len(q.AcceleratorTypes) > 0 {
		result["accelerator_types"] = strings.Join(q.AcceleratorTypes, ",")
	}

	// 排序字段
	if q.OrderByCreatedAt != 0 {
		result["sort_by_created_at"] = strconv.Itoa(q.OrderByCreatedAt)
	}

	if q.OrderByPriority != 0 {
		result["sort_by_priority"] = strconv.Itoa(q.OrderByPriority)
	}

	return result
}

type ListMatchResponse struct {
	Total int64                     `json:"total"`
	PN    int64                     `json:"pn"`
	RN    int64                     `json:"rn"`
	Data  []*MatchingDetailResponse `json:"data"`
}

type GetStatisticsResponse struct {
	Statistics map[string]int64 `json:"statistics"`
}

type ListFilterOptionsResponse struct {
	AcceleratorTypes []string         `json:"accelerator_types"`
	BabiUnits        []BabiUnitOption `json:"babi_units"`
	DisplayStatuses  []string         `json:"display_statuses"`
}

type BabiUnitOption struct {
	BabiUnit     string   `json:"babi_unit"`
	SubBabiUnits []string `json:"sub_babi_units"`
}

type MatchingIntentRequest struct {
	Name           string          `json:"name"`
	Description    string          `json:"description"`
	MatchingIntent *MatchingIntent `json:"intent"`
	IdempotencyKey string          `json:"idempotencyKey"`
}

func (i *MatchingIntent) Validate() ([]string, error) {
	var warnings = []string{}

	if i == nil {
		return warnings, fmt.Errorf("matching intent is required")
	}
	requester := i.Requester
	if requester == nil {
		return warnings, fmt.Errorf(".requester is required")
	}

	if requester.BusinessLineId == "" {
		return warnings, fmt.Errorf(".requester.businessLineId is required")
	}

	if requester.ResourceGroupId == "" {
		return warnings, fmt.Errorf(".requester.resourceGroupId is required")
	}

	if requester.BusinessLineName == "" {
		return warnings, fmt.Errorf(".requester.businessLineName is required")
	}

	if i.TimeWindow == nil {
		return warnings, fmt.Errorf(".timeWindow is required")
	}

	if i.TimeWindow.StartTime.IsZero() {
		return warnings, fmt.Errorf(".timeWindow.startTime is required")
	}

	if i.TimeWindow.EndTime == nil || i.TimeWindow.EndTime.IsZero() {
		return warnings, fmt.Errorf(".timeWindow.endTime is required")
	}

	minTime := time.Now().UTC().Add(-time.Hour)
	if i.TimeWindow.StartTime.Before(minTime) {
		return warnings, fmt.Errorf(".timeWindow.startTime must after %v", minTime)
	}

	if (*i.TimeWindow.EndTime).Before(minTime) {
		return warnings, fmt.Errorf(".timeWindow.endTime must after %v", minTime)
	}

	maxTime := time.Now().UTC().Add(time.Hour * 24 * 30)
	if i.TimeWindow.StartTime.After(maxTime) {
		return warnings, fmt.Errorf(".timeWindow.startTime must before %v", maxTime)
	}

	if (*i.TimeWindow.EndTime).After(maxTime) {
		return warnings, fmt.Errorf(".timeWindow.endTime must before %v", maxTime)
	}

	windowDuration := i.TimeWindow.Duration()
	if windowDuration <= 0 {
		return warnings, fmt.Errorf(".timeWindow.endTime must be after .timeWindow.startTime")
	}

	minDuration := i.GetMinDuration()
	maxDuration := i.GetMaxDuration()
	if i.TimeWindow.FlexibleAllocation != nil &&
		i.TimeWindow.FlexibleAllocation.MinDuration != nil &&
		minDuration <= 0 {
		return warnings, fmt.Errorf("minDuration must be positive")
	}
	if i.TimeWindow.FlexibleAllocation != nil &&
		i.TimeWindow.FlexibleAllocation.MaxDuration != nil &&
		maxDuration <= 0 {
		return warnings, fmt.Errorf("maxDuration must be positive")
	}
	if minDuration > 0 && windowDuration < time.Duration(minDuration)*time.Hour {
		return warnings, fmt.Errorf("duration between .timeWindow.startTime and .timeWindow.endTime must be at least minDuration %v", minDuration)
	}
	if maxDuration > 0 && windowDuration < time.Duration(maxDuration)*time.Hour {
		return warnings, fmt.Errorf("duration between .timeWindow.startTime and .timeWindow.endTime must be at least maxDuration %v", maxDuration)
	}
	if maxDuration > 0 && minDuration > maxDuration {
		return warnings, fmt.Errorf("minDuration %d must not exceed maxDuration %d", minDuration, maxDuration)
	}

	if i.Groups == nil {
		return warnings, fmt.Errorf(".groups[0] is required")
	}
	groups := *i.Groups
	if len(groups) == 0 {
		return warnings, fmt.Errorf(".groups[0] is required")
	}

	for index, x := range groups {
		if x.AcceleratorPreference.PreferredTypes == nil {
			return warnings, fmt.Errorf(".groups[%d].acceleratorPreference.preferredTypes is required", index)
		}

		if len(*x.AcceleratorPreference.PreferredTypes) == 0 {
			return warnings, fmt.Errorf(".groups[%d].acceleratorPreference.preferredTypes is required", index)
		}

		if x.Elasticity != nil {
			if utils.IntPtrToIntOrZero(x.Elasticity.MinReplicas) == 0 {
				return warnings, fmt.Errorf(".groups[%d].minReplicas must be larger than 0", index)
			}

			if utils.IntPtrToIntOrZero(x.Elasticity.MaxReplicas) == 0 {
				return warnings, fmt.Errorf(".groups[%d].maxReplicas is required", index)
			}
			if utils.IntPtrToIntOrZero(x.Elasticity.MaxReplicas) < utils.IntPtrToIntOrZero(x.Elasticity.MinReplicas) {
				return warnings, fmt.Errorf(".groups[%d].maxReplicas must be larger than .groups[%d].minReplicas", index, index)
			}
		} else {
			if utils.IntPtrToIntOrZero(x.Replicas) == 0 {
				return warnings, fmt.Errorf(".groups[%d].replicas is required", index)
			}
		}

		if x.LocationConstraint == nil {
			return warnings, fmt.Errorf(".groups[%d].locationConstraint is required", index)
		}

		if x.LocationConstraint.Dc == nil {
			return warnings, fmt.Errorf(".groups[%d].locationConstraint.dc is required", index)
		}

		if x.LocationConstraint.Zone == nil {
			return warnings, fmt.Errorf(".groups[%d].locationConstraint.zone is required", index)
		}

		if x.LocationConstraint.Cluster == nil {
			return warnings, fmt.Errorf(".groups[%d].locationConstraint.cluster is required", index)
		}

		{
			dcMaxLocations := x.LocationConstraint.Dc.MaxLocations
			zoneMaxLocations := x.LocationConstraint.Zone.MaxLocations
			clusterMaxLocations := x.LocationConstraint.Cluster.MaxLocations
			if clusterMaxLocations != nil || dcMaxLocations != nil || zoneMaxLocations != nil {
				if !x.IsSameNodeLevel() {
					return warnings, fmt.Errorf("only supports sameNodeLevel (max locations = 1 in zone, dc, cluster)")
				}
			}
		}

		if x.VolcConfig != nil {
			if x.AcceleratorPreference.PreferredTypes != nil {
				if len(*x.AcceleratorPreference.PreferredTypes) == 0 {
					warnings = append(warnings, fmt.Sprintf(".groups[%d].acceleratorPreference.preferredTypes is required", index))
					return warnings, fmt.Errorf(".groups[%d].acceleratorPreference.preferredTypes is required", index)
				}
				if len(*x.AcceleratorPreference.PreferredTypes) > 1 {
					warnings = append(warnings, fmt.Sprintf(".groups[%d].acceleratorPreference.preferredTypes must be one", index))
					return warnings, fmt.Errorf(".groups[%d].acceleratorPreference.preferredTypes must be one", index)
				}
			}
		}
	}

	if slices.Contains([]FlexibleAllocationPriority{FlexibleAllocationPriorityEarliest, FlexibleAllocationPriorityLatest}, i.GetFlexibleAllocationPriority()) {
		if !i.IsSameNodeLevel() {
			return warnings, fmt.Errorf(".timeWindow.flexibleAllocation.priority must be any or empty when maxLocations != 1")
		}
	}

	return warnings, nil
}

func (i *MatchingIntent) GetMinDuration() int64 {
	if i.TimeWindow == nil {
		return 0
	}

	if i.TimeWindow.FlexibleAllocation == nil {
		return 0
	}

	if i.TimeWindow.FlexibleAllocation.MinDuration == nil {
		return 0
	}

	return int64(*i.TimeWindow.FlexibleAllocation.MinDuration)
}

func (i *MatchingIntent) GetMaxDuration() int64 {
	if i.TimeWindow == nil {
		return 0
	}

	if i.TimeWindow.FlexibleAllocation == nil {
		return 0
	}

	if i.TimeWindow.FlexibleAllocation.MaxDuration == nil {
		return 0
	}

	return int64(*i.TimeWindow.FlexibleAllocation.MaxDuration)
}

func (i *MatchingIntent) IsSameNodeLevel() bool {
	if i == nil || i.Groups == nil {
		return false
	}
	var isSameNodeLevel = true
	for _, x := range *i.Groups {
		if !x.IsSameNodeLevel() {
			isSameNodeLevel = false
			break
		}
	}
	return isSameNodeLevel
}

func (i *MatchingIntent) GetFlexibleAllocationPriority() FlexibleAllocationPriority {
	if i == nil || i.TimeWindow == nil || i.TimeWindow.FlexibleAllocation == nil ||
		i.TimeWindow.FlexibleAllocation.Priority == nil {
		return FlexibleAllocationPriorityAny
	}
	return *i.TimeWindow.FlexibleAllocation.Priority
}
