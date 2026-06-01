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

// In-process mock TCE catalog. Pairs with tceMockProvisioner to let the
// console boot in environments where the real TCE control plane is
// unreachable. Returns empty discovery results — the planner's submit
// path doesn't depend on the catalog, so empty responses are sufficient
// for demo deployments.

package tcemock

import (
	"context"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/catalog"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
)

// tceMockCatalog implements catalog.Catalog with hard-coded empty
// responses. Sufficient for demo deployments where catalog data isn't
// needed for the primary planner submit flow.
type tceMockCatalog struct{}

// newCatalog creates a new TCE mock catalog.
func newCatalog() (catalog.Catalog, error) {
	return &tceMockCatalog{}, nil
}

// Provider returns the provider type.
func (c *tceMockCatalog) Provider() types.ResourceProvisionType {
	return types.ResourceProvisionTypeTCEMock
}

// ListRegions lists available regions for the catalog.
func (c *tceMockCatalog) ListRegions(ctx context.Context) ([]types.RegionSpec, error) {
	return []types.RegionSpec{}, nil
}

// ListInstanceTypes lists available instance types in the given region.
func (c *tceMockCatalog) ListInstanceTypes(ctx context.Context, region *types.RegionSpec) ([]types.InstanceTypeSpec, error) {
	return []types.InstanceTypeSpec{}, nil
}

// ListResources lists resources matching the given options.
func (c *tceMockCatalog) ListResources(ctx context.Context, opts *catalog.ResourceListOptions) ([]catalog.Resource, error) {
	return []catalog.Resource{}, nil
}

// ListResourcePredictions lists predicted resources keyed by identifier.
func (c *tceMockCatalog) ListResourcePredictions(ctx context.Context, opts *catalog.ResourceListOptions) (map[string]catalog.Resource, error) {
	return map[string]catalog.Resource{}, nil
}

// ListPricing lists pricing information matching the given options.
func (c *tceMockCatalog) ListPricing(ctx context.Context, opts *catalog.ResourceListOptions) ([]catalog.ResourcePricing, error) {
	return []catalog.ResourcePricing{}, nil
}

// ListPricingPredictions lists predicted pricing keyed by identifier.
func (c *tceMockCatalog) ListPricingPredictions(ctx context.Context, opts *catalog.ResourceListOptions) (map[string]catalog.ResourcePricing, error) {
	return map[string]catalog.ResourcePricing{}, nil
}
