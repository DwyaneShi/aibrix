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

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/http"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/json"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/jwt"
)

const headerKeyJWTToken = "X-JWT-TOKEN"

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

func (impl ClientImpl) GetTicketPriority(ctx context.Context, ticketID int64) (*TicketPriorityResult, error) {
	req := http.NewPost(impl.getFullUrlPath("/api/orders/ticketPriority")).WithBody(&TicketPriorityRequest{TicketID: ticketID})
	bodyBytes, err := impl.sendRequest(req)
	if err != nil {
		return nil, err
	}

	var resp TicketPriorityResponse
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if !strings.EqualFold(resp.Code, "success") {
		return nil, fmt.Errorf("GetTicketPriority failed: code=%s", resp.Code)
	}
	if resp.Data == nil {
		return nil, fmt.Errorf("GetTicketPriority failed: data is nil")
	}
	return resp.Data, nil
}

func (impl ClientImpl) GetOrderTimeline(ctx context.Context, matchID string) ([]OrderTimelineEntry, error) {
	req := http.NewGet(impl.getFullUrlPath(fmt.Sprintf("/api/orders/timeline/%s", matchID)))
	bodyBytes, err := impl.sendRequest(req)
	if err != nil {
		return nil, err
	}

	var resp OrderTimelineResponse
	if err := impl.jsonParser.Unmarshal(bodyBytes, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}
	if !strings.EqualFold(resp.Code, "success") {
		return nil, fmt.Errorf("GetOrderTimeline failed: code=%s", resp.Code)
	}
	return resp.Data, nil
}
