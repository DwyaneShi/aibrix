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

package types

import "fmt"

// Only used for extending the custom region specs.
type ExtensionRegionSpecs struct {
	// TCE contains TCE-specific region information.
	TCE *TCERegion `json:"tce,omitempty"`
}

func (r *ExtensionRegionSpecs) GetRegion() Region {
	if r.TCE != nil {
		return r.TCE
	}
	return nil
}

// TCERegion contains TCE-specific region information.
type TCERegion struct {
	// Zone is the geographic zone (e.g., "CN", "US").
	Zone string `json:"zone,omitempty"`

	// Dc is the datacenter (e.g., "LF", "HL").
	Dc string `json:"dc,omitempty"`

	// PhysicalCluster is the physical cluster name.
	PhysicalCluster string `json:"physicalCluster,omitempty"`

	// LogicalCluster is the logical cluster name.
	LogicalCluster string `json:"logicalCluster,omitempty"`
}

func (r *TCERegion) String() string {
	return fmt.Sprintf("%s/%s/%s/%s", orNone(r.Zone), orNone(r.Dc), orNone(r.PhysicalCluster), orNone(r.LogicalCluster))
}

type TCERegionAffinity struct {
	Zone RegionAffinity `json:"zone"`
	Dc   RegionAffinity `json:"dc"`
	// Cluster format: "Federation/scheduled-benchmark"
	Cluster RegionAffinity `json:"cluster"`
}

// NewTCERegion creates a RegionSpec for TCE.
func NewTCERegion(zone, dc, physicalCluster, logicalCluster string) *RegionSpec {
	spec := &RegionSpec{
		ExtensionRegionSpecs: ExtensionRegionSpecs{
			TCE: &TCERegion{Zone: zone, Dc: dc, PhysicalCluster: physicalCluster, LogicalCluster: logicalCluster},
		},
	}
	return spec
}
