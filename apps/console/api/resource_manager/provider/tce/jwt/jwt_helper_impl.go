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
package jwt

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	rmhttp "github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/http"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/json"
)

var _ JwtHelper = (*jwtHelperImpl)(nil)

type JwtHelper interface {
	GenJwtToken(ctx context.Context) (string, error)
	GetUsername(ctx context.Context, jwtToken string) (string, error)
}

type jwtHelperImpl struct {
	secret    string
	urlPrefix string

	httpClient rmhttp.Client
	mutex      sync.RWMutex
	jwtToken   string
	expiresAt  time.Time
}

const (
	jwtTokenHeader      = "X-Jwt-Token"
	authTokenPath       = "/api/v1/jwt"
	tokenCacheDuration  = 10 * time.Minute
	authorizationHeader = "Authorization"
)

func NewJwtHelper(secret, urlPrefix string) JwtHelper {
	return &jwtHelperImpl{
		secret:     secret,
		urlPrefix:  urlPrefix,
		httpClient: rmhttp.NetClient,
	}
}

func (j *jwtHelperImpl) GenJwtToken(ctx context.Context) (string, error) {
	_ = ctx

	j.mutex.RLock()
	if j.jwtToken != "" && time.Now().UTC().Before(j.expiresAt) {
		token := j.jwtToken
		j.mutex.RUnlock()
		return token, nil
	}
	j.mutex.RUnlock()

	j.mutex.Lock()
	defer j.mutex.Unlock()
	if j.jwtToken != "" && time.Now().UTC().Before(j.expiresAt) {
		return j.jwtToken, nil
	}

	token, err := j.fetchJwtToken()
	if err != nil {
		return "", err
	}
	j.jwtToken = token
	j.expiresAt = time.Now().UTC().Add(tokenCacheDuration)
	return token, nil
}

func (j *jwtHelperImpl) fetchJwtToken() (string, error) {
	req := rmhttp.NewGet(j.urlPrefix + authTokenPath).WithHeaders(map[string]string{
		authorizationHeader: "Bearer " + j.secret,
	})
	resp, err := j.httpClient.SendRequest(req)
	if err != nil {
		return "", fmt.Errorf("request auth token failed: %w", err)
	}
	if resp.GetStatusCode() != rmhttp.StatusOK {
		return "", fmt.Errorf("request auth token failed: status=%d", resp.GetStatusCode())
	}

	token := strings.TrimSpace(resp.GetHeader(jwtTokenHeader))
	if token == "" {
		return "", errors.New("empty X-Jwt-Token from auth response")
	}
	return token, nil
}

func (j *jwtHelperImpl) GetUsername(ctx context.Context, jwtStr string) (string, error) {
	_ = ctx

	parts := strings.Split(jwtStr, ".")
	if len(parts) < 2 {
		return "", errors.New("invalid jwt token")
	}

	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("decode jwt payload failed: %w", err)
	}

	var claims struct {
		Username string `json:"username"`
		Sub      string `json:"sub"`
	}
	if err := json.Unmarshal(payloadBytes, &claims); err != nil {
		return "", fmt.Errorf("unmarshal jwt payload failed: %w", err)
	}

	if claims.Username != "" {
		return claims.Username, nil
	}
	if claims.Sub != "" {
		return claims.Sub, nil
	}
	return "", errors.New("username not found in jwt claims")
}
