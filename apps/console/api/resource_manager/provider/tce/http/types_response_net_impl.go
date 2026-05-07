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
	"fmt"
	"io"
	netHttp "net/http"

	"k8s.io/klog/v2"
)

type ResponseNetImpl struct {
	resp *netHttp.Response
	body []byte
}

func newResponseNetImpl(resp *netHttp.Response) (*ResponseNetImpl, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		if body != nil {
			klog.Warningf("newResponseNetImpl failed to read response body:%s statusCode:%d", string(body), resp.StatusCode)
		}
		return nil, fmt.Errorf("generate response err: %v", err)
	}

	return &ResponseNetImpl{
		resp: resp,
		body: body,
	}, nil
}

func (r *ResponseNetImpl) GetBody() []byte {
	if r == nil {
		return []byte{}
	}
	return r.body
}

func (r *ResponseNetImpl) GetHeader(key string) string {
	return r.resp.Header.Get(key)
}

func (r *ResponseNetImpl) GetStatusCode() int {
	return r.resp.StatusCode
}
