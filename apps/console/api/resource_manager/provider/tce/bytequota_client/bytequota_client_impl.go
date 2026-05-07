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

package bytequota_client

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client/resource_pool_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/http"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/json"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/jwt"
)

type ClientImpl struct {
	api        string
	jwtHelper  jwt.JwtHelper
	httpClient http.Client
	jsonParser json.Parser
}

func NewClientImpl(api string, jwtHelper jwt.JwtHelper) *ClientImpl {
	return &ClientImpl{
		api:        api,
		jwtHelper:  jwtHelper,
		httpClient: http.NetClient,
		jsonParser: json.DefaultParser,
	}
}

const headerKeyJWTToken = "X-JWT-TOKEN"

func (impl ClientImpl) getHeaders() (map[string]string, error) {
	platformJwtToken, err := impl.jwtHelper.GenJwtToken(context.Background())
	if err != nil {
		return nil, err
	}

	return map[string]string{
		headerKeyJWTToken: platformJwtToken,
		"Content-type":    "application/json",
	}, nil
}

func (impl ClientImpl) getFullUrlPath(path string) string {
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return fmt.Sprintf("%s%s", impl.api, path)
}

func (impl *ClientImpl) sendRequest(req *http.Request) ([]byte, error) {
	headers, err := impl.getHeaders()
	if err != nil {
		return nil, fmt.Errorf("get request headers failed: %w", err)
	}
	resp, err := impl.httpClient.SendRequest(req.WithHeaders(headers))
	if err != nil {
		return nil, fmt.Errorf("send request failed: %w", err)
	}
	return resp.GetBody(), nil
}

func (impl ClientImpl) ListResourcePools(ctx context.Context, query *resource_pool_types.ListResourcePoolsRequest) ([]*resource_pool_types.ResourcePoolResp, error) {
	req := http.NewGet(impl.getFullUrlPath("/api/v2/resource-pools"))
	if query != nil {
		req.WithQueryParam(query.GetParams())
	}
	bodyBytes, err := impl.sendRequest(req)
	if err != nil {
		return nil, err
	}

	var resp struct {
		Result    []*resource_pool_types.ResourcePoolResp `json:"data,omitempty"`
		ErrorCode int                                     `json:"error_code,omitempty"`
		Message   string                                  `json:"message,omitempty"`
	}
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if resp.ErrorCode != 0 {
		message := "unknown"
		if resp.Message != "" {
			message = resp.Message
		}
		return nil, fmt.Errorf("ListResourcePools failed: code=%d error=%s", resp.ErrorCode, message)
	}
	if resp.Result == nil {
		return nil, nil
	}
	return resp.Result, nil
}

func (impl ClientImpl) GetBusinessLineInfo(ctx context.Context, psm, platform string) (*BusinessLineInfo, error) {
	query := &resource_pool_types.ListResourcePoolsRequest{
		Psm:      &psm,
		Platform: &platform,
	}
	resourcePools, err := impl.ListResourcePools(ctx, query)
	if err != nil {
		return nil, err
	}

	if len(resourcePools) == 0 {
		return nil, fmt.Errorf("no resource pool found for psm=%s, platform=%s", psm, platform)
	}

	candidateMap := map[string]BusinessLineInfo{}
	for _, rp := range resourcePools {
		if rp.ResourceGroup == nil || rp.ResourceGroup.ResourceGroup == nil {
			continue
		}
		group := rp.ResourceGroup.ResourceGroup
		businessLineId := ""
		businessLineName := ""
		if len(group.SharingScopes) > 0 {
			scope := group.SharingScopes[len(group.SharingScopes)-1]
			businessLineId = fmt.Sprintf("%d", scope.ID)
			businessLineName = scope.Name
		} else {
			businessLineId = group.ResourceGroupMeta.ID
			businessLineName = group.ResourceGroupMeta.Name
		}
		info := BusinessLineInfo{
			BusinessLineId:   businessLineId,
			BusinessLineName: businessLineName,
			ResourceGroupId:  group.ResourceGroupMeta.ID,
		}
		key := info.BusinessLineId + "|" + info.ResourceGroupId
		candidateMap[key] = info
	}
	candidates := make([]BusinessLineInfo, 0, len(candidateMap))
	for _, v := range candidateMap {
		candidates = append(candidates, v)
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no business line found for psm=%s, platform=%s", psm, platform)
	}
	// order candidate by BusinessLineName in descending order of length.
	sort.Slice(candidates, func(i, j int) bool {
		return len(candidates[i].BusinessLineName) > len(candidates[j].BusinessLineName)
	})
	selected := candidates[0]
	result := &BusinessLineInfo{
		BusinessLineId:           selected.BusinessLineId,
		BusinessLineName:         selected.BusinessLineName,
		ResourceGroupId:          selected.ResourceGroupId,
		BusinessLineDependencies: &candidates,
	}
	return result, nil
}

// GetBusinessLineInfoV2 retrieves business line info using the nearest resource group API.
func (impl ClientImpl) GetBusinessLineInfoV2(ctx context.Context, psm string) (*BusinessLineInfo, error) {
	// Call the nearest resource group API
	reqBody := resource_pool_types.GetResourceGroupRequest{
		SupplyDomains:             []string{"*/*/*"},
		OnlyFederatedGPUPartition: true,
		QueueName:                 "default",
	}

	req := http.NewPost(impl.getFullUrlPath(fmt.Sprintf("/api/v2/resource-groups/nearest_group/%s", psm))).WithBody(reqBody)
	bodyBytes, err := impl.sendRequest(req)
	if err != nil {
		return nil, fmt.Errorf("send request failed: %w", err)
	}

	var resp resource_pool_types.GetResourceGroupResponse
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}

	if resp.Message == "" || strings.ToLower(resp.Message) != "success" {
		return nil, fmt.Errorf("GetBusinessLineInfoV2 failed: message=%s", resp.Message)
	}

	// Extract business line info from sharing_scopes
	var finalNodeID uint64
	var finalBusinessLineName string
	finalResourceGroupID := resp.Data.ID

	if len(resp.Data.SharingScopes) > 0 {
		if len(resp.Data.SharingScopes) == 1 {
			finalNodeID = resp.Data.SharingScopes[0].ID
			finalBusinessLineName = resp.Data.SharingScopes[0].Name
		} else {
			// Sort by ID and pick the smallest one
			sortedScopes := make([]struct {
				ID   uint64
				Name string
			}, len(resp.Data.SharingScopes))
			for i, scope := range resp.Data.SharingScopes {
				sortedScopes[i].ID = scope.ID
				sortedScopes[i].Name = scope.Name
			}
			sort.Slice(sortedScopes, func(i, j int) bool {
				return sortedScopes[i].ID < sortedScopes[j].ID
			})
			finalNodeID = sortedScopes[0].ID
			finalBusinessLineName = sortedScopes[0].Name
		}
	}

	result := &BusinessLineInfo{
		BusinessLineId:   fmt.Sprintf("%d", finalNodeID),
		BusinessLineName: finalBusinessLineName,
		ResourceGroupId:  finalResourceGroupID,
	}

	return result, nil
}
