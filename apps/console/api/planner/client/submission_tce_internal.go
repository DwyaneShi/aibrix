/*
Copyright 2026 The Aibrix Team.

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

// TCEResourceAllocation is the TCE backend's ResourceAllocation implementation.
// It must live in package client because ResourceAllocation is sealed by the
// unexported isResourceAllocation marker; only types declared here can satisfy it.

package client

// TCEResourceAllocation is carried on AIBrixExtraBody.ResourceAllocation. It
// extends the provision id with the cluster-scoped allocation detail TCE returns.
type TCEResourceAllocation struct {
	ProvisionID               string            `json:"provision_id,omitempty"`
	ProvisionResourceDeadline int64             `json:"provision_resource_deadline,omitempty"`
	ResourceDetails           []ResourceDetails `json:"resource_details,omitempty"`
}

func (*TCEResourceAllocation) isResourceAllocation() {}

// ResourceDetails describes a cluster-scoped allocation.
type ResourceDetails struct {
	Provider         string `json:"provider,omitempty"`
	EndpointCluster  string `json:"endpoint_cluster,omitempty"`
	ResourcePoolName string `json:"resource_pool_name,omitempty"`
	SaleMode         string `json:"salemode,omitempty"`
	QoSLevel         string `json:"qos_level,omitempty"`
	LogicalCluster   string `json:"logical_cluster,omitempty"`

	Resources []ResourceItem `json:"resources,omitempty"`
}

// ResourceItem describes one role's accelerator request.
type ResourceItem struct {
	AcceleratorType     string `json:"accelerator_type,omitempty"`
	AcceleratorCategory string `json:"accelerator_category,omitempty"`
	CPU                 string `json:"cpu,omitempty"`
	Memory              string `json:"memory,omitempty"`
	AcceleratorCount    int    `json:"accelerator_count,omitempty"`
	Replica             int    `json:"replica,omitempty"`
	// Name is the role name within the StormService (defaults to "default"
	// today; reserved for future per-role differentiation).
	Name string `json:"name,omitempty"`
}
