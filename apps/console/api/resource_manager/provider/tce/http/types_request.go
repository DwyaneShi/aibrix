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

import "context"

type Request struct {
	ctx         context.Context
	method      string
	url         string
	body        interface{}
	queryParams map[string]string
	headers     map[string]string
}

func NewPost(url string) *Request {
	return newRequest(MethodPost, url)
}

func NewGet(url string) *Request {
	return newRequest(MethodGet, url)
}

func NewPut(url string) *Request {
	return newRequest(MethodPut, url)
}

func NewPatch(url string) *Request {
	return newRequest(MethodPatch, url)
}

func NewDelete(url string) *Request {
	return newRequest(MethodDelete, url)
}

func newRequest(method, url string) *Request {
	return &Request{
		ctx:         context.Background(),
		method:      method,
		url:         url,
		body:        nil,
		queryParams: nil,
		headers:     nil,
	}
}

func (r *Request) WithBody(body interface{}) *Request {
	r.body = body
	return r
}

func (r *Request) WithContext(ctx context.Context) *Request {
	if ctx == nil {
		ctx = context.Background()
	}
	r.ctx = ctx
	return r
}

func (r *Request) WithQueryParam(queryParams map[string]string) *Request {
	r.queryParams = queryParams
	return r
}

func (r *Request) WithHeaders(headers map[string]string) *Request {
	r.headers = headers
	return r
}

func (r *Request) Method() string {
	return r.method
}

func (r *Request) Context() context.Context {
	if r.ctx == nil {
		return context.Background()
	}
	return r.ctx
}

func (r *Request) Url() string {
	return r.url
}

func (r *Request) Body() interface{} {
	return r.body
}

func (r *Request) QueryParams() map[string]string {
	return r.queryParams
}

func (r *Request) Headers() map[string]string {
	return r.headers
}
