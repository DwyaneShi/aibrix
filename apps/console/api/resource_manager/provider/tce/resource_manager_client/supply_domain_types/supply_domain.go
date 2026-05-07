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
	"fmt"
	"time"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client/quota_types"
	quota_supply_domain_types "github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client/supply_domain_types"
)

const (
	ZONE             = "zone"
	PHYSICAL_CLUSTER = "physical_cluster"
	LOGICAL_CLUSTER  = "logical_cluster"
	DC               = "dc"
	PARTITION        = "partition"
)

type GetSupplyDomainsRequest struct {
	Platform    *string
	ConvertUnit *bool
	Clusters    *string
	Detail      *bool
}

type SupplyDomainResp struct {
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	IsDeleted bool      `json:"is_deleted"`
	quota_supply_domain_types.SupplyDomainMeta
	quota_supply_domain_types.SupplyDomainSpec
	convertedUnit bool `json:"converted_unit"`
}

func (p *GetSupplyDomainsRequest) GetParams() map[string]string {
	params := make(map[string]string)
	if p.Platform != nil {
		params["platform"] = *p.Platform
	}
	if p.Clusters != nil {
		params["clusters"] = *p.Clusters
	}
	if p.ConvertUnit != nil {
		params["convert_unit"] = "true"
	}
	if p.Detail != nil {
		params["detail"] = "true"
	}
	return params
}

func (sd *SupplyDomainResp) CacheConvertedUnit(convertedUnit bool) {
	sd.convertedUnit = convertedUnit
}

func (sd SupplyDomainResp) GetZone() string {
	return sd.Labels[ZONE]
}

func (sd SupplyDomainResp) GetPhysicalCluster() string {
	return sd.Labels[PHYSICAL_CLUSTER]
}

func (sd SupplyDomainResp) GetLogicalCluster() string {
	return sd.Labels[LOGICAL_CLUSTER]
}

func (sd SupplyDomainResp) GetVirtualCluster() string {
	return fmt.Sprintf("%s/%s", sd.GetPhysicalCluster(), sd.GetLogicalCluster())
}

func (sd SupplyDomainResp) GetDc() string {
	return sd.Labels[DC]
}

func (sd SupplyDomainResp) GetPartition() string {
	return sd.Labels[PARTITION]
}

func (sd SupplyDomainResp) IsSocketOnly() bool {
	return sd.Features.UseSocket
}

func (sd SupplyDomainResp) IsSmallContainerOnly() bool {
	return !sd.Features.UseSocket && !sd.Features.IsIntegrated
}

func (sd SupplyDomainResp) IsMixContainer() bool {
	return sd.Features.IsIntegrated
}

func (sd SupplyDomainResp) SupportSocket() bool {
	return sd.IsSocketOnly() || sd.IsMixContainer()
}

func (sd SupplyDomainResp) SupportSmallContainer() bool {
	return sd.IsSmallContainerOnly() || sd.IsMixContainer()
}

func (sd SupplyDomainResp) IsFreezed() bool {
	return sd.Freezed
}

func (sd SupplyDomainResp) GetPackages() []string {
	packageList := make([]string, 0, len(sd.Package.Supply))
	for packageName := range sd.Package.Supply {
		packageList = append(packageList, packageName)
	}
	return packageList
}

func (sd SupplyDomainResp) GetSupplyQuota() *quota_types.Quota {
	return quota_types.NewResourceQuota(sd.Quota.Supply, sd.convertedUnit)
}

func (sd SupplyDomainResp) GetAllocatableQuota() *quota_types.Quota {
	return quota_types.NewResourceQuota(sd.Quota.Allocatable, sd.convertedUnit)
}

func (sd SupplyDomainResp) GetAllocatablePackageNum(packageName string) int64 {
	return sd.Package.Allocatable[packageName]
}

func (sd SupplyDomainResp) GetSupplyPackageNum(packageName string) int64 {
	return sd.Package.Supply[packageName]
}

func (sd SupplyDomainResp) CheckAllocatableQuota(requestQuota *quota_types.Quota) (passCheck bool, quotaGap *quota_types.Quota) {
	if !requestQuota.HasPositiveValue() {
		return true, nil
	}
	isSocketOnly := sd.IsSocketOnly()
	remainQuota := sd.GetAllocatableQuota().GetPart(isSocketOnly)
	requestQuota = requestQuota.GetPart(isSocketOnly)
	quotaGap = remainQuota.Reduce(requestQuota)

	return !quotaGap.HasNegativeValue(), quotaGap.GetNegativePart().Negate()
}

func (sd SupplyDomainResp) CheckAllocatablePackage(requestPackage map[string]int64) (passCheck bool, packageGap map[string]int64) {
	passCheck = true
	packageGap = make(map[string]int64)
	for name, num := range requestPackage {
		gap := sd.GetAllocatablePackageNum(name) - num
		if gap < 0 {
			passCheck = false
			packageGap[name] = -gap
		}
	}

	return passCheck, packageGap
}

func (sd SupplyDomainResp) GetAllocatablePackage(packageName string) int64 {
	return sd.Package.Allocatable[packageName]
}
