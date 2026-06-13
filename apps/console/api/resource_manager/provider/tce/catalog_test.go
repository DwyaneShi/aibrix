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

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/catalog"
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
