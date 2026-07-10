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
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/config"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/credential"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/gpu_center_client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/jwt"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client"

	"k8s.io/klog/v2"
)

const (
	// TODO: change to "AIBrix" once matching_api support it
	AIBrixPlatformName                 = "Bernard"
	AIBrixPlatformPSM                  = "inf.aibrix.platform"
	AIBrixPlatformBusinessLinePlatform = "Compute"
)

var AIBrixPlatformServiceAccount = credential.TcePlatformServiceAccount

type tceClientset struct {
	RegionConfig          *config.RegionConfig           `json:"regionConfig"`
	ResourceManagerClient resource_manager_client.Client `json:"-"`
	ByteQuotaClient       bytequota_client.Client        `json:"-"`
	GPUCenterClient       gpu_center_client.Client       `json:"-"`
}

func newTCEClientset() (*tceClientset, error) {
	regionConfig := config.GetRegionConfig()
	klog.Infof("regionConfig: %v", regionConfig)

	jwtHelper := jwt.NewJwtHelper(AIBrixPlatformServiceAccount, regionConfig.JwtAuthUrlPrefix)
	resourceManagerClient := resource_manager_client.NewClientImpl(regionConfig.ResourceManagerAPI, jwtHelper)
	byteQuotaClient := bytequota_client.NewClientImpl(regionConfig.ByteQuotaAPI, jwtHelper)

	var gpuCenterClient gpu_center_client.Client
	if regionConfig.GPUCenterAPI != "" {
		gpuCenterClient = gpu_center_client.NewClientImpl(regionConfig.GPUCenterAPI, jwtHelper)
	}
	return &tceClientset{
		RegionConfig:          regionConfig,
		ResourceManagerClient: resourceManagerClient,
		ByteQuotaClient:       byteQuotaClient,
		GPUCenterClient:       gpuCenterClient,
	}, nil
}
