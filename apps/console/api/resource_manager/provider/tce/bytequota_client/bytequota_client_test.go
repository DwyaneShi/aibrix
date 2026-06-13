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
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client/resource_pool_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/credential"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/jwt"
)

var (
	TestingJwtAuthUrlPrefix = "https://cloud.bytedance.net/auth"
	TestingByteQuotaAPI     = "http://tce-planning.byted.org"
)

var TestingServiceAccount = credential.TcePlatformServiceAccount

func setupClient(t *testing.T) *ClientImpl {
	t.Helper()
	j := jwt.NewJwtHelper(TestingServiceAccount, TestingJwtAuthUrlPrefix)
	return NewClientImpl(TestingByteQuotaAPI, j)
}

func TestListResourcePools(t *testing.T) {
	client := setupClient(t)
	ctx := context.Background()

	t.Run("print response", func(t *testing.T) {
		start := time.Now().UTC()
		psm := "inf.aibrix.platform"
		platform := "Compute"
		query := &resource_pool_types.ListResourcePoolsRequest{
			Psm:      &psm,
			Platform: &platform,
		}
		resp, err := client.ListResourcePools(ctx, query)
		if err != nil {
			t.Fatalf("ListResourcePools failed: %v", err)
		}
		if len(resp) == 0 {
			t.Fatalf("ListResourcePools failed: resp is empty")
		}
		first := resp[0]
		b, _ := json.Marshal(first)
		fmt.Printf("[ListResourcePools] elapsed=%s resp=%s\n", time.Since(start), string(b))
	})
}

func TestGetBusinessLineInfo(t *testing.T) {
	client := setupClient(t)
	ctx := context.Background()

	t.Run("print response", func(t *testing.T) {
		start := time.Now().UTC()
		psm := "inf.aibrix.platform"
		platform := "Compute"
		resp, err := client.GetBusinessLineInfo(ctx, psm, platform)
		if err != nil {
			t.Fatalf("GetBusinessLineInfo failed: %v", err)
		}
		if resp == nil {
			t.Fatalf("GetBusinessLineInfo failed: resp is nil")
		}
		b, _ := json.Marshal(resp)
		fmt.Printf("[GetBusinessLineInfo] elapsed=%s resp=%s\n", time.Since(start), string(b))
	})
}
