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
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/credential"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/jwt"
	supplydomaintypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/supply_domain_types"
)

var (
	TestingJwtAuthUrlPrefix   = "https://cloud.bytedance.net/auth"
	TestingResourceManagerAPI = "https://resource-manager.byted.org"
)

var TestingServiceAccount = credential.TcePlatformServiceAccount

func setupClient(t *testing.T) *ClientImpl {
	t.Helper()
	j := jwt.NewJwtHelper(TestingServiceAccount, TestingJwtAuthUrlPrefix)
	return NewClientImpl(TestingResourceManagerAPI, j)
}

func TestListScheduledMatch(t *testing.T) {
	client := setupClient(t)
	ctx := context.Background()

	t.Run("print response", func(t *testing.T) {
		start := time.Now()
		resp, err := client.ListScheduledMatch(ctx, nil)
		if err != nil {
			t.Fatalf("ListScheduledMatch failed: %v", err)
		}
		b, _ := json.Marshal(resp)
		fmt.Printf("[ListScheduledMatch] elapsed=%s resp=%s\n", time.Since(start), string(b))
	})
}

func TestGetStatistics(t *testing.T) {
	client := setupClient(t)
	ctx := context.Background()

	t.Run("print response", func(t *testing.T) {
		start := time.Now()
		resp, err := client.GetStatistics(ctx, nil)
		if err != nil {
			t.Fatalf("GetStatistics failed: %v", err)
		}
		b, _ := json.Marshal(resp)
		fmt.Printf("[GetStatistics] elapsed=%s resp=%s\n", time.Since(start), string(b))
	})
}

func TestGetSupplyDomains(t *testing.T) {
	client := setupClient(t)
	ctx := context.Background()

	t.Run("print response", func(t *testing.T) {
		start := time.Now()
		platform := "Compute"
		detail := true
		convertUnit := true
		query := &supplydomaintypes.GetSupplyDomainsRequest{
			Platform:    &platform,
			Detail:      &detail,
			ConvertUnit: &convertUnit,
		}
		resp, err := client.GetSupplyDomains(ctx, query)
		if err != nil {
			t.Fatalf("GetSupplyDomains failed: %v", err)
		}
		if resp == nil {
			t.Fatalf("GetSupplyDomains failed: resp is nil")
		}
		b, _ := json.Marshal(resp)
		fmt.Printf("[GetSupplyDomains] elapsed=%s resp=%s\n", time.Since(start), string(b))
	})
}
