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

package supply_controller_types

import (
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
)

type Filters struct {
	ResourceSelector ResourceSelector `json:"resource_selector"`
	Effective        Effective        `json:"effective"`
	Options          Options          `json:"options"`
}

type Target struct {
	DC      string `json:"dc"`
	Cluster string `json:"cluster"`
}

type ResourceSelector struct {
	MatchLabels      map[string]string `json:"matchLabels"`
	MatchExpressions []MatchExpression `json:"matchExpressions"`
}

type MatchExpression struct {
	Key      string   `json:"key"`
	Operator string   `json:"operator"`
	Values   []string `json:"values"`
}

type Effective struct {
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
}

type Options struct {
	HostExclusive      bool `json:"hostExclusive"`
	IsVirtualNodeLevel bool `json:"isVirtualNodeLevel"`
}

type ListResourceRequestFilters struct {
	ResourceSelector ResourceSelector `json:"resourceSelector"`
	TopologyKeys     []string         `json:"topologyKeys"`
	Options          Options          `json:"options"`
}

type ListResourceRequestFilter struct {
	Filters ListResourceRequestFilters `json:"filters"`
}

type ListResourceRequest struct {
	Target  Target                    `json:"target"`
	Request ListResourceRequestFilter `json:"request"`
}

type ClusterResources []struct {
	Cluster         string          `json:"cluster"`
	ClusterResource ClusterResource `json:"clusterResource"`
}

type ResourceItem map[string]map[string]resource.Quantity

type TimeResourceItem map[string]ResourceItem

type ResourceStat struct {
	Allocated              ResourceItem     `json:"allocated"`
	Suppy                  ResourceItem     `json:"supply"`
	Allocatable            ResourceItem     `json:"allocatable"`
	TimeSpecifiedAllocated TimeResourceItem `json:"timeSpecifiedAllocated"`
	TimeSpecifiedAvailable TimeResourceItem `json:"timeSpecifiedAvailable"`
	TimeSpecifiedSupply    TimeResourceItem `json:"timeSpecifiedSupply"`
}

type ClusterResourceItem struct {
	HierarchyStats []ClusterResourceItem `json:"hierarchyStats"`
	Key            string                `json:"key"`
	Value          string                `json:"value"`
	ResourceStat   ResourceStat          `json:"resourceStat"`
}

type ClusterResource struct {
	MemberOverviewResource map[string][]ClusterResourceItem `json:"memberOverviewResource"`
	OverviewResource       []ClusterResourceItem            `json:"overviewResource"`
}

type ScalarQuotaRequest struct {
	Target Target `json:"target"`
}
