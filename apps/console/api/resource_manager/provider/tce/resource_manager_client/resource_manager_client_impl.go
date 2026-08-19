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
	"fmt"
	"strings"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/http"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/json"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/jwt"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/scheduled_plan_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/supply_domain_types"
	"k8s.io/klog/v2"
)

type ClientImpl struct {
	api        string
	jwtHelper  jwt.JwtHelper
	httpClient http.Client
	jsonParser json.Parser
}

type ResourceManagerResponseBase struct {
	StatusCode    int    `json:"status_code"` // api error code
	StatusMessage string `json:"status_msg"`  // api error message
}

func NewClientImpl(api string, jwtHelper jwt.JwtHelper) *ClientImpl {
	return &ClientImpl{
		api:        api,
		jwtHelper:  jwtHelper,
		httpClient: http.NetClient,
		jsonParser: json.DefaultParser,
	}
}

func (impl ClientImpl) getHeaders(ctx context.Context) (map[string]string, error) {
	platformJwtToken, err := impl.jwtHelper.GenJwtToken(ctx)
	if err != nil {
		return nil, err
	}

	return map[string]string{
		HeaderKeyJWTToken: platformJwtToken,
		"Content-type":    "application/json",
	}, nil
}

func (impl ClientImpl) getFullUrlPath(path string) string {
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return fmt.Sprintf("%s%s", impl.api, path)
}

func (impl *ClientImpl) sendRequest(ctx context.Context, req *http.Request) ([]byte, error) {
	headers, err := impl.getHeaders(ctx)
	if err != nil {
		return nil, fmt.Errorf("get request headers failed: %w", err)
	}
	for key, value := range req.Headers() {
		headers[key] = value
	}
	resp, err := impl.httpClient.SendRequest(req.WithContext(ctx).WithHeaders(headers))
	if err != nil {
		return nil, fmt.Errorf("send request failed: %w", err)
	}
	return resp.GetBody(), nil
}

func (impl ClientImpl) GetQuotaView(ctx context.Context, query *scheduled_plan_types.QuotaViewReq) ([]*scheduled_plan_types.QuotaViewItem, error) {
	req := http.NewGet(impl.getFullUrlPath("/resource_manager/matching_api/v1/resource/quota_view"))
	if query != nil {
		req = req.WithQueryParam(query.GetParams())
	}
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result []*scheduled_plan_types.QuotaViewItem `json:"result"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("GetQuotaView failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("GetQuotaView failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) GetScheduledMatch(ctx context.Context, id string) (*scheduled_plan_types.MatchingResult, error) {
	req := http.NewGet(impl.getFullUrlPath(fmt.Sprintf("/resource_manager/matching_api/v1/match/%s", id)))
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.MatchingResult `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("GetScheduledMatch failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("GetScheduledMatch failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) GetScheduledMatchDetail(ctx context.Context, id string) (*scheduled_plan_types.MatchingDetailResponse, error) {
	req := http.NewGet(impl.getFullUrlPath(fmt.Sprintf("/resource_manager/matching_api/v1/match/%s/detail", id)))
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.MatchingDetailResponse `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("GetScheduledMatchDetail failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("GetScheduledMatchDetail failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) CancelScheduledMatch(ctx context.Context, id string) (*scheduled_plan_types.MatchingResult, error) {
	req := http.NewPost(impl.getFullUrlPath(fmt.Sprintf("/resource_manager/matching_api/v1/match/%s/cancel", id)))
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.MatchingResult `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("CancelScheduledMatch failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("CancelScheduledMatch failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) ListScheduledMatch(ctx context.Context, query *scheduled_plan_types.ListMatchQuery) (*scheduled_plan_types.ListMatchResponse, error) {
	req := http.NewGet(impl.getFullUrlPath("/resource_manager/matching_api/v1/match"))
	if query != nil {
		req = req.WithQueryParam(query.GetParams())
	}
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.ListMatchResponse `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("ListScheduledMatch failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("ListScheduledMatch failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) GetStatistics(ctx context.Context, query *scheduled_plan_types.ListMatchQuery) (*scheduled_plan_types.GetStatisticsResponse, error) {
	req := http.NewGet(impl.getFullUrlPath("/resource_manager/matching_api/v1/match/statistics"))
	if query != nil {
		req = req.WithQueryParam(query.GetParams())
	}
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.GetStatisticsResponse `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("GetStatistics failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("GetStatistics failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) ListFilterOptions(ctx context.Context) (*scheduled_plan_types.ListFilterOptionsResponse, error) {
	req := http.NewGet(impl.getFullUrlPath("/resource_manager/matching_api/v1/match/filter_options"))
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.ListFilterOptionsResponse `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("ListFilterOptions failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("ListFilterOptions failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) CreateScheduledMatch(ctx context.Context, query *scheduled_plan_types.MatchingIntentRequest) (*scheduled_plan_types.MatchingResult, error) {
	if query == nil {
		return nil, fmt.Errorf("query is nil")
	}
	if query.MatchingIntent == nil {
		return nil, fmt.Errorf("query.MatchingIntent is nil")
	}

	warnings, err := query.MatchingIntent.Validate()
	if len(warnings) > 0 {
		for _, warning := range warnings {
			klog.Warningf("CreateScheduledMatch %s", warning)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("CreateScheduledMatch failed: %w", err)
	}

	req := http.NewPost(impl.getFullUrlPath("/resource_manager/matching_api/v1/match")).
		WithBody(query)
	if strings.TrimSpace(query.IdempotencyKey) != "" {
		req = req.WithHeaders(map[string]string{HeaderKeyIdempotencyKey: query.IdempotencyKey})
	}
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.MatchingResult `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("CreateScheduledMatch failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("CreateScheduledMatch failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) UpdateScheduledMatch(ctx context.Context, id string, query *scheduled_plan_types.MatchingIntentRequest) (*scheduled_plan_types.MatchingResult, error) {
	req := http.NewPut(impl.getFullUrlPath(fmt.Sprintf("/resource_manager/matching_api/v1/match/%s", id)))
	if query != nil {
		req = req.WithBody(query)
	}
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.MatchingResult `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("UpdateScheduledMatch failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("UpdateScheduledMatch failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) CommitScheduledMatch(ctx context.Context, id string) (*scheduled_plan_types.MatchingResult, error) {
	req := http.NewPost(impl.getFullUrlPath(fmt.Sprintf("/resource_manager/matching_api/v1/match/%s/commit", id)))
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result *scheduled_plan_types.MatchingResult `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("CommitScheduledMatch failed: code=%d error=%s", resp.StatusCode, message)
	}
	if resp.Result == nil {
		return nil, fmt.Errorf("CommitScheduledMatch failed: result is nil")
	}
	return resp.Result, nil
}

func (impl ClientImpl) GetSupplyDomains(ctx context.Context, req *supply_domain_types.GetSupplyDomainsRequest) ([]*supply_domain_types.SupplyDomainResp, error) {
	httpReq := http.NewGet(impl.getFullUrlPath("/resource_manager/proxy/supply_domains"))
	if req != nil {
		httpReq = httpReq.WithQueryParam(req.GetParams())
	}
	bodyBytes, err := impl.sendRequest(ctx, httpReq)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result []*supply_domain_types.SupplyDomainResp `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("GetSupplyDomains failed: code=%d error=%s", resp.StatusCode, message)
	}

	return resp.Result, nil
}

func (impl ClientImpl) GetMatchTimeline(ctx context.Context, matchID string) ([]scheduled_plan_types.MatchTimelineEntry, error) {
	req := http.NewGet(impl.getFullUrlPath(fmt.Sprintf("/resource_manager/matching_api/v1/match/%s/timeline", matchID)))
	bodyBytes, err := impl.sendRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ResourceManagerResponseBase
		Result []scheduled_plan_types.MatchTimelineEntry `json:"result,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.StatusCode != 0 {
		message := resp.StatusMessage
		return nil, fmt.Errorf("GetMatchTimeline failed: code=%d error=%s", resp.StatusCode, message)
	}

	return resp.Result, nil
}
