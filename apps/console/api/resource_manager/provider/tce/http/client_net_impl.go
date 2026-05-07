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
package http

import (
	"bytes"
	"fmt"
	netHttp "net/http"
	netUrl "net/url"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/json"
	"k8s.io/klog/v2"
)

// ClientNetImpl 基于net/http库封装的http_client。
// 一般情况下不建议使用这个，更建议使用ClientHertzImpl
type ClientNetImpl struct {
}

func NewClientNetImpl() *ClientNetImpl {
	return &ClientNetImpl{}
}

func (h ClientNetImpl) SendRequest(request *Request) (Response, error) {
	var (
		url       = request.Url()
		dataBytes []byte
		err       error
	)

	if len(request.QueryParams()) > 0 {
		v := netUrl.Values{}
		for key, value := range request.QueryParams() {
			v.Add(key, value)
		}
		url = fmt.Sprintf("%s?%s", request.Url(), v.Encode())
	}

	body := request.Body()
	if body != nil {
		dataBytes, err = json.Marshal(body)
		if err != nil {
			return nil, err
		}
	}

	klog.V(4).Infof("request: %s %s %v", request.method, url, dataBytes)
	return h.httpCon(request.Method(), url, dataBytes, request.Headers())
}

func (h ClientNetImpl) httpCon(method, url string, data []byte, headers map[string]string) (response *ResponseNetImpl, err error) {
	req, err := netHttp.NewRequest(method, url, bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	client := &netHttp.Client{}
	httpResp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() {
		if httpResp != nil {
			httpResp.Body.Close()
		}
	}()
	return newResponseNetImpl(httpResp)
}
