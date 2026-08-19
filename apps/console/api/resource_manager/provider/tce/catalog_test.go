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

package tce

import (
	"context"
	"testing"
	"time"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/catalog"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/scheduled_plan_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/utils"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
)

func TestTCECatalog_ListRegions(t *testing.T) {
	cat, err := newCatalog()
	if err != nil {
		t.Fatalf("newCatalog failed: %v", err)
	}
	regions, err := cat.ListRegions(context.Background())
	if err != nil {
		t.Fatalf("ListRegions failed: %v", err)
	}
	if len(regions) == 0 {
		t.Fatalf("ListRegions failed: no regions")
	}
	for _, region := range regions {
		t.Logf("region: %v", region.GetRegion())
	}
}

func TestTCECatalog_ListResources(t *testing.T) {
	cat, err := newCatalog()
	if err != nil {
		t.Fatalf("newCatalog failed: %v", err)
	}
	resources, err := cat.ListResources(context.Background(), &catalog.ResourceListOptions{
		Region: types.RegionSpec{
			ExtensionRegionSpecs: types.ExtensionRegionSpecs{
				TCE: &types.TCERegion{
					Zone: "China-North",
					Dc:   "LF",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("ListResources failed: %v", err)
	}
	if len(resources) == 0 {
		t.Fatalf("ListResources failed: no resources")
	}
	for _, resource := range resources {
		t.Logf("resource: region=%v, overview_count=%d, %s", resource.Region.GetRegion(), len(resource.Overview), utils.Marshal(resource))
	}
}

func TestTCECatalog_ListResourcePredictions(t *testing.T) {
	controller := gomock.NewController(t)
	defer controller.Finish()

	resourceManagerClient := resource_manager_client.NewMockClient(controller)
	startTime := time.Date(2026, 8, 20, 10, 0, 0, 0, time.UTC)
	endTime := startTime.Add(time.Hour)
	predictedSupply := int64(120)
	predictedAllocatable := int64(95)

	resourceManagerClient.EXPECT().
		GetQuotaView(gomock.Any(), gomock.Any()).
		DoAndReturn(func(ctx context.Context, request *scheduled_plan_types.QuotaViewReq) ([]*scheduled_plan_types.QuotaViewItem, error) {
			assert.Equal(t, startTime, request.StartTime)
			assert.Equal(t, endTime, request.EndTime)
			assert.Equal(t, []string{"China-North"}, request.Zones)
			assert.Equal(t, []string{"LF"}, request.Dcs)
			assert.Equal(t, []string{"Federation/xpu"}, request.Clusters)
			return []*scheduled_plan_types.QuotaViewItem{
				{
					StartTime:                    startTime.Format(time.RFC3339),
					EndTime:                      endTime.Format(time.RFC3339),
					Zone:                         "China-North",
					Dc:                           "LF",
					Partition:                    "gpu",
					PhysicalCluster:              "Federation",
					LogicalCluster:               "xpu",
					HardwareType:                 "xpu",
					HardwareKind:                 "NVIDIA-H20",
					HardwareSupply:               100,
					HardwareAllocatable:          70,
					HardwareSupplyPredicted:      &predictedSupply,
					HardwareAllocatablePredicted: &predictedAllocatable,
				},
			}, nil
		})

	tceCatalog := &tceCatalog{
		clientset: &tceClientset{ResourceManagerClient: resourceManagerClient},
	}

	predictions, err := tceCatalog.ListResourcePredictions(context.Background(), &catalog.ResourceListOptions{
		Region: types.RegionSpec{
			ExtensionRegionSpecs: types.ExtensionRegionSpecs{
				TCE: &types.TCERegion{
					Zone:            "China-North",
					Dc:              "LF",
					PhysicalCluster: "Federation",
					LogicalCluster:  "xpu",
				},
			},
		},
		StartTime: &startTime,
		EndTime:   &endTime,
	})

	require.NoError(t, err)
	require.Len(t, predictions, 1)
	resource, ok := predictions["China-North/LF/Federation/xpu"]
	require.True(t, ok)
	require.Len(t, resource.Overview, 1)
	scheduled := resource.Overview[0].Stat.Scheduled
	require.NotNil(t, scheduled)
	assert.Equal(t, "120", scheduled.Supply[startTime.Format(time.RFC3339)]["xpu"]["NVIDIA-H20"])
	assert.Equal(t, "95", scheduled.Allocatable[startTime.Format(time.RFC3339)]["xpu"]["NVIDIA-H20"])
	assert.Empty(t, scheduled.Allocated)
}
