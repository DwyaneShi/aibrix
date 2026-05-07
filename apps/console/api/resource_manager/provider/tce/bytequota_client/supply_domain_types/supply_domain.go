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

package supply_domain_types

import (
	"time"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client/resource_pool_types"
)

const (
	Platform        = "platform"
	Zone            = "zone"
	DC              = "dc"
	PhysicalCluster = "physical_cluster"
	LogicalCluster  = "logical_cluster"
	ResourceLevel   = "resource_level"
	Partition       = "partition"
	Compute         = "compute"
	Guarantee       = "guaranteed"
	MicroPartition  = "micro"
	Federation      = "Federation"
)

type ListSupplyDomainsParam struct {
	Platform        *string
	Zone            *string
	Dc              *string
	PhysicalCluster *string
	LogicalCluster  *string
	ResourceLevel   *string
	Detail          *bool
	ConvertUnit     *bool
	Partition       *string
}

func NewListSupplyDomainsParam() *ListSupplyDomainsParam {
	compute := Compute
	guarantee := Guarantee
	return &ListSupplyDomainsParam{
		Platform:      &compute,
		ResourceLevel: &guarantee,
	}
}

func (p *ListSupplyDomainsParam) GetParams() map[string]string {
	params := make(map[string]string)
	if p.Platform != nil {
		params[Platform] = *p.Platform
	}
	if p.Zone != nil {
		params[Zone] = *p.Zone
	}
	if p.PhysicalCluster != nil {
		params[PhysicalCluster] = *p.PhysicalCluster
	}
	if p.LogicalCluster != nil {
		params[LogicalCluster] = *p.LogicalCluster
	}
	if p.Dc != nil {
		params[DC] = *p.Dc
	}
	if p.ResourceLevel != nil {
		params[ResourceLevel] = *p.ResourceLevel
	}
	if p.Detail != nil {
		params["detail"] = "true"
	}
	if p.ConvertUnit != nil {
		params["convert_unit"] = "true"
	}
	if p.Partition != nil && p.PhysicalCluster != nil && *p.PhysicalCluster == Federation {
		params[Partition] = *p.Partition
	}
	return params
}

func (p *ListSupplyDomainsParam) WithZone(zone string) *ListSupplyDomainsParam {
	p.Zone = &zone
	return p
}

func (p *ListSupplyDomainsParam) WithDc(dc string) *ListSupplyDomainsParam {
	p.Dc = &dc
	return p
}

func (p *ListSupplyDomainsParam) WithPhysicalCluster(pc string) *ListSupplyDomainsParam {
	p.PhysicalCluster = &pc
	return p
}

func (p *ListSupplyDomainsParam) WithLogicalCluster(lc string) *ListSupplyDomainsParam {
	p.LogicalCluster = &lc
	return p
}

func (p *ListSupplyDomainsParam) WithDetail(detail bool) *ListSupplyDomainsParam {
	p.Detail = &detail
	return p
}

func (p *ListSupplyDomainsParam) WithConvertUnit(convertUnit bool) *ListSupplyDomainsParam {
	p.ConvertUnit = &convertUnit
	return p
}

func (p *ListSupplyDomainsParam) WithPartition(partition string) *ListSupplyDomainsParam {
	p.Partition = &partition
	return p
}

func (p *ListSupplyDomainsParam) GetConvertUnit() bool {
	if p.ConvertUnit == nil {
		return false
	}
	return *p.ConvertUnit
}

func (p *ListSupplyDomainsParam) GetPartition() *string {
	return p.Partition
}

type SupplyDomainResp struct {
	RecordMeta       SupplyDomainRecordMeta `json:"record_meta"`
	SupplyDomainMeta `json:"supply_domain_meta"`
	SupplyDomainSpec `json:"supply_domain_spec"`
	convertedUnit    bool `json:"-"`
}

func (sd *SupplyDomainResp) CacheConvertedUnit(convertedUnit bool) {
	sd.convertedUnit = convertedUnit
}

func (sd SupplyDomainResp) GetZone() string {
	return sd.Labels[Zone]
}

func (sd SupplyDomainResp) GetDc() string {
	return sd.Labels[DC]
}

func (sd SupplyDomainResp) GetPhysicalCluster() string {
	return sd.Labels[PhysicalCluster]
}

func (sd SupplyDomainResp) GetLogicalCluster() string {
	return sd.Labels[LogicalCluster]
}

func (sd SupplyDomainResp) GetPartition() string {
	return sd.Labels[Partition]
}

func (sd SupplyDomainResp) IsSocketOnly() bool {
	return sd.Features.UseSocket
}

func (sd SupplyDomainResp) IsFreezed() bool {
	return sd.Freezed
}

type SupplyDomainRecordMeta struct {
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	IsDeleted bool      `json:"is_deleted"`
}

type SupplyDomainMeta struct {
	Name        string                             `json:"name"`
	Platform    string                             `json:"platform"`
	Labels      map[string]string                  `json:"labels"`
	Annotations map[string]string                  `json:"annotations"`
	Buffer      resource_pool_types.ResourceItem   `json:"buffer"`
	Freezed     bool                               `json:"freezed"`
	Features    resource_pool_types.DomainFeatures `json:"features"`
	Admins      []string                           `json:"admins"`
}

type SupplyDomainSpec struct {
	Quota   resource_pool_types.QuotaSupply   `json:"quota"`
	Package resource_pool_types.PackageSupply `json:"package"`
}

type GetSupplyDomainListResponse struct {
	Code    int                 `json:"error_code"`
	Message string              `json:"message"`
	Data    []*SupplyDomainResp `json:"data"`
}

func (r GetSupplyDomainListResponse) CacheConvertedUnit(convertedUnit bool) {
	for _, d := range r.Data {
		d.CacheConvertedUnit(convertedUnit)
	}
}

func (r GetSupplyDomainListResponse) GetData() []*SupplyDomainResp {
	return r.Data
}
