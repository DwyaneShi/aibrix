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
	"strings"
	"time"
)

type QuotaViewReq struct {
	StartTime     time.Time `json:"start_time"`    // 格式 2026-03-07T03:29:00Z， 需要换算成 utc 时间
	EndTime       time.Time `json:"end_time"`      // 格式 2026-03-07T03:29:00Z
	Zones         []string  `json:"zone"`          // 地区 ,如果是多个值，则用逗号拼接，如 China-East,China-North
	Dcs           []string  `json:"dc"`            // 如果是多个值，则用逗号拼接，如 GL,HJ
	Clusters      []string  `json:"clusters"`      // 如果是多个值，则用逗号拼接，如 CloudNative/ai,Candy/dandelion-ai-mix
	HardwareKinds []string  `json:"hardware_kind"` // 如果是多个值，则用逗号拼接，如 NVIDIA-L20,NVIDIA-L40
}

func (p *QuotaViewReq) GetParams() map[string]string {
	params := make(map[string]string)
	params["start_time"] = p.StartTime.Format("2006-01-02T15:04:05Z")
	params["end_time"] = p.EndTime.Format("2006-01-02T15:04:05Z")
	if p.Zones != nil {
		params["zone"] = strings.Join(p.Zones, ",")
	}
	if p.Dcs != nil {
		params["dc"] = strings.Join(p.Dcs, ",")
	}
	if p.Clusters != nil {
		params["clusters"] = strings.Join(p.Clusters, ",")
	}
	if p.HardwareKinds != nil {
		params["hardware_kind"] = strings.Join(p.HardwareKinds, ",")
	}
	return params
}

type QuotaViewItem struct {
	StartTime               string `json:"start_time"`
	EndTime                 string `json:"end_time"`
	Zone                    string `json:"zone"`
	Dc                      string `json:"dc"`
	Partition               string `json:"partition"`
	PhysicalCluster         string `json:"physical_cluster"`
	LogicalCluster          string `json:"logical_cluster"`
	HardwareType            string `json:"hardware_type"`
	HardwareKind            string `json:"hardware_kind"`
	HardwareSupply          int64  `json:"hardware_supply"`
	HardwareAllocatable     int64  `json:"hardware_allocatable"`
	HardwareAllocated       int64  `json:"hardware_allocated"`
	HardwareAllocatedRate   int64  `json:"hardware_allocated_rate"`
	HardwareBooked          int64  `json:"hardware_booked"`
	HardwareBookedRate      int64  `json:"hardware_booked_rate"`
	HardwareAheadBooked     int64  `json:"hardware_ahead_booked"`
	HardwareAheadBookedRate int64  `json:"hardware_ahead_booked_rate"`
	HardwareUsage           int64  `json:"hardware_usage"`
	HardwareAvailable       int64  `json:"hardware_available"`
	// 预测数据，仅在查询未来时间段的 quota 数据时返回。
	HardwareSupplyPredicted      *int64 `json:"hardware_supply_predicted,omitempty"`
	HardwareAllocatablePredicted *int64 `json:"hardware_allocatable_predicted,omitempty"`
}
