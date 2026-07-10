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

package gpu_center_client

type TicketPriorityRequest struct {
	TicketID int64 `json:"ticket_id"`
}

type TicketPriorityResponse struct {
	Code string                `json:"code"`
	Data *TicketPriorityResult `json:"data,omitempty"`
}

type TicketPriorityResult struct {
	Priority              int64   `json:"priority"`
	ResourceGroupPriority int64   `json:"resource_group_priority"`
	ResourceGroupWeight   float64 `json:"resource_group_weight"`
	GPUUtilPriority       int64   `json:"gpu_util_priority"`
	GPUUtilWeight         float64 `json:"gpu_util_weight"`
	BizPriority           int64   `json:"biz_priority"`
	BizWeight             float64 `json:"biz_weight"`
	WorkloadPriority      int64   `json:"workload_priority"`
	WorkloadWeight        float64 `json:"workload_weight"`
	SceneWeight           float64 `json:"scene_weight"`
	PlatformWeight        float64 `json:"platform_weight"`
}
