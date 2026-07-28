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

package resource_manager_client

import (
	"context"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/scheduled_plan_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/supply_domain_types"
)

//go:generate mockgen -source=resource_manager_client.go -destination=./resource_manager_client_mock.go -package=resource_manager_client

type Client interface {
	// GetQuotaView 获取实时的quota 视图 .
	// 数据分为 历史数据和未来数据，需要对查询时间进行判断
	// 1. 如果 endTime < now ，则从历史数据中查
	// 2. 如果 startTime > time.Now().Truncate(time.Hour) ，则从未来数据中查
	// 3. 否则需要对查询日期进行拆分，拆分成两个时间段，分别从历史数据和未来数据中查 [start, time.Now().Truncate(time.Hour)) 和 (time.Now().Truncate(time.Hour), end]
	GetQuotaView(ctx context.Context, req *scheduled_plan_types.QuotaViewReq) ([]*scheduled_plan_types.QuotaViewItem, error)
	GetScheduledMatch(ctx context.Context, id string) (*scheduled_plan_types.MatchingResult, error)
	GetScheduledMatchDetail(ctx context.Context, id string) (*scheduled_plan_types.MatchingDetailResponse, error)
	CancelScheduledMatch(ctx context.Context, id string) (*scheduled_plan_types.MatchingResult, error)
	ListScheduledMatch(ctx context.Context, query *scheduled_plan_types.ListMatchQuery) (*scheduled_plan_types.ListMatchResponse, error)
	GetStatistics(ctx context.Context, query *scheduled_plan_types.ListMatchQuery) (*scheduled_plan_types.GetStatisticsResponse, error)
	ListFilterOptions(ctx context.Context) (*scheduled_plan_types.ListFilterOptionsResponse, error)
	CreateScheduledMatch(ctx context.Context, query *scheduled_plan_types.MatchingIntentRequest) (*scheduled_plan_types.MatchingResult, error)
	UpdateScheduledMatch(ctx context.Context, id string, query *scheduled_plan_types.MatchingIntentRequest) (*scheduled_plan_types.MatchingResult, error)
	CommitScheduledMatch(ctx context.Context, id string) (*scheduled_plan_types.MatchingResult, error)
	GetSupplyDomains(ctx context.Context, req *supply_domain_types.GetSupplyDomainsRequest) ([]*supply_domain_types.SupplyDomainResp, error)
	GetMatchTimeline(ctx context.Context, matchID string) ([]scheduled_plan_types.MatchTimelineEntry, error)
}
