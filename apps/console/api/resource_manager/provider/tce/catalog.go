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
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/catalog"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/scheduled_plan_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/supply_domain_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
)

const regionsCacheTTL = 1 * time.Hour

// tceCatalog implements catalog.Catalog for TCE.
type tceCatalog struct {
	clientset *tceClientset

	// regions cache
	regionsCache      []types.RegionSpec
	regionsCacheTime  time.Time
	regionsCacheMutex sync.RWMutex
}

// newCatalog creates a new TCE catalog.
func newCatalog() (catalog.Catalog, error) {
	clientset, err := newTCEClientset()
	if err != nil {
		return nil, err
	}
	return &tceCatalog{
		clientset: clientset,
	}, nil
}

// Provider returns the provider type.
func (c *tceCatalog) Provider() types.ResourceProvisionType {
	return types.ResourceProvisionTypeTCE
}

// ListRegions lists available regions for the catalog.
func (c *tceCatalog) ListRegions(ctx context.Context) ([]types.RegionSpec, error) {
	// Check cache first
	c.regionsCacheMutex.RLock()
	if c.regionsCache != nil && time.Since(c.regionsCacheTime) < regionsCacheTTL {
		defer c.regionsCacheMutex.RUnlock()
		return c.regionsCache, nil
	}
	c.regionsCacheMutex.RUnlock()

	// Fetch from API
	platform := AIBrixPlatformName
	param := supply_domain_types.GetSupplyDomainsRequest{
		Platform: &platform,
	}
	domains, err := c.clientset.ResourceManagerClient.GetSupplyDomains(ctx, &param)
	if err != nil {
		return nil, err
	}

	regionMap := make(map[string]types.RegionSpec)
	for _, domain := range domains {
		zone := domain.GetZone()
		dc := domain.GetDc()
		physicalCluster := domain.GetPhysicalCluster()
		logicalCluster := domain.GetLogicalCluster()

		key := zone + "/" + dc + "/" + physicalCluster + "/" + logicalCluster
		if _, exists := regionMap[key]; !exists {
			regionMap[key] = types.RegionSpec{
				ExtensionRegionSpecs: types.ExtensionRegionSpecs{
					TCE: &types.TCERegion{
						Zone:            zone,
						Dc:              dc,
						PhysicalCluster: physicalCluster,
						LogicalCluster:  logicalCluster,
					},
				},
			}
		}
	}

	regions := make([]types.RegionSpec, 0, len(regionMap))
	for _, region := range regionMap {
		regions = append(regions, region)
	}

	// Update cache
	c.regionsCacheMutex.Lock()
	c.regionsCache = regions
	c.regionsCacheTime = time.Now().UTC()
	c.regionsCacheMutex.Unlock()

	return regions, nil
}

// ListInstanceTypes lists available instance types for the catalog.
func (c *tceCatalog) ListInstanceTypes(ctx context.Context, region *types.RegionSpec) ([]types.InstanceTypeSpec, error) {
	return nil, types.ErrNotImplemented
}

// ListResources lists available resources matching the options.
// NOTE: don't support S2/S1 resource views yet.
//
// Example returns:
//
//		{
//		  "region":{
//		     "tce":{
//		        "zone":"China-North",
//		        "dc":"LF",
//		        "physicalCluster":"Federation",
//		        "logicalCluster":"xpu"
//		     }
//		  },
//		  "overview":[
//		     {
//		        "key":"partition",
//		        "value":"gpu",
//		        "stat":{
//		           "scheduled":{
//		              "allocated":{
//		                 "2026-05-07T02:00:00Z":{
//		                    "xpu":{
//		                       "jiuhuashan-96T":"52"
//		                    }
//		                 }
//		              },
//		              "supply":{
//		                 "2026-05-07T02:00:00Z":{
//		                    "xpu":{
//		                       "jiuhuashan-96T":"1780"
//		                    }
//		                 }
//		              },
//		              "allocatable":{
//		                 "2026-05-07T02:00:00Z":{
//		                    "xpu":{
//		                       "jiuhuashan-96T":"1728"
//		                    }
//		                 }
//		              }
//		           }
//		        }
//		     },
//		     {
//	         ...
//		     }
//		  ],
//		  "provider":"tce"
//		}
func (c *tceCatalog) ListResources(ctx context.Context, opts *catalog.ResourceListOptions) ([]catalog.Resource, error) {
	req := &scheduled_plan_types.QuotaViewReq{}

	if opts != nil && opts.StartTime != nil {
		req.StartTime = opts.StartTime.UTC().Truncate(time.Hour)
	} else {
		req.StartTime = time.Now().UTC().Truncate(time.Hour)
	}

	if opts != nil && opts.EndTime != nil {
		req.EndTime = opts.EndTime.UTC().Truncate(time.Hour)
	} else {
		req.EndTime = req.StartTime.Add(time.Hour)
	}

	if req.EndTime.Before(req.StartTime) {
		return nil, fmt.Errorf("end time must be after start time")
	}

	if opts != nil && opts.Region.TCE != nil {
		if opts.Region.TCE.Zone != "" {
			req.Zones = []string{opts.Region.TCE.Zone}
		}
		if opts.Region.TCE.Dc != "" {
			req.Dcs = []string{opts.Region.TCE.Dc}
		}
		if opts.Region.TCE.PhysicalCluster != "" || opts.Region.TCE.LogicalCluster != "" {
			cluster := opts.Region.TCE.PhysicalCluster
			if opts.Region.TCE.LogicalCluster != "" {
				cluster += "/" + opts.Region.TCE.LogicalCluster
			}
			req.Clusters = []string{cluster}
		}
	}

	items, err := c.clientset.ResourceManagerClient.GetQuotaView(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("get quota view: %w", err)
	}

	regionMap := make(map[string]*catalog.Resource)
	for _, item := range items {
		regionKey := item.Zone + "/" + item.Dc + "/" + item.PhysicalCluster + "/" + item.LogicalCluster
		region := types.NewTCERegion(item.Zone, item.Dc, item.PhysicalCluster, item.LogicalCluster)

		resource, exists := regionMap[regionKey]
		if !exists {
			resource = &catalog.Resource{
				Provider: types.ResourceProvisionTypeTCE,
				RegionResource: catalog.RegionResource{
					Region:   region,
					Overview: []catalog.RegionResourceItem{},
				},
			}
			regionMap[regionKey] = resource
		}

		overview := resource.Overview
		overview = append(overview, buildRegionResourceItem(item))
		resource.Overview = overview
	}

	resources := make([]catalog.Resource, 0, len(regionMap))
	for _, r := range regionMap {
		resources = append(resources, *r)
	}

	return resources, nil
}

func buildRegionResourceItem(item *scheduled_plan_types.QuotaViewItem) catalog.RegionResourceItem {
	resourceType := item.HardwareType
	resourceName := item.HardwareKind
	timestamp := item.StartTime

	qtyAllocated := fmt.Sprintf("%d", item.HardwareAllocated)
	qtySupply := fmt.Sprintf("%d", item.HardwareSupply)
	qtyAllocatable := fmt.Sprintf("%d", item.HardwareAllocatable)

	allocated := catalog.ScheduledResourceItem{
		timestamp: catalog.ResourceItem{
			resourceType: map[string]string{resourceName: qtyAllocated},
		},
	}
	supply := catalog.ScheduledResourceItem{
		timestamp: catalog.ResourceItem{
			resourceType: map[string]string{resourceName: qtySupply},
		},
	}
	allocatable := catalog.ScheduledResourceItem{
		timestamp: catalog.ResourceItem{
			resourceType: map[string]string{resourceName: qtyAllocatable},
		},
	}

	return catalog.RegionResourceItem{
		Key:   "partition",
		Value: item.Partition,
		Stat: catalog.ResourceStat{
			Scheduled: &catalog.ScheduledResourceStatItem{
				Allocated:   allocated,
				Supply:      supply,
				Allocatable: allocatable,
			},
		},
	}
}

// ListResourcePredictions lists resource predictions for the options.
func (c *tceCatalog) ListResourcePredictions(ctx context.Context, opts *catalog.ResourceListOptions) (map[string]catalog.Resource, error) {
	return nil, types.ErrNotImplemented
}

// ============================================================================
// PricingCatalog Implementation
// ============================================================================

// ListPricing returns pricing information for instance types.
func (c *tceCatalog) ListPricing(ctx context.Context, opts *catalog.ResourceListOptions) ([]catalog.ResourcePricing, error) {
	return nil, types.ErrNotImplemented
}

// ListPricingPredictions lists pricing predictions for the options.
func (c *tceCatalog) ListPricingPredictions(ctx context.Context, opts *catalog.ResourceListOptions) (map[string]catalog.ResourcePricing, error) {
	return nil, types.ErrNotImplemented
}
