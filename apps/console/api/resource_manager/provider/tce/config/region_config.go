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

package config

import "os"

type AppRegion string

const (
	AppRegionProdCN     AppRegion = "prod_cn"
	AppRegionProdUS     AppRegion = "prod_us"
	AppRegionProdI18N   AppRegion = "prod_i18n"
	AppRegionProdI18NBD AppRegion = "prod_i18nbd"
	AppRegionProdTTP    AppRegion = "prod_ttp"
	AppRegionTTP        AppRegion = "ttp"
	AppRegionBoe        AppRegion = "boe"
	AppRegionTesting    AppRegion = "testing"
	AppRegionSandbox    AppRegion = "sandbox"
)

var (
	AppRegionEnvs = []string{
		"APP_REGION",
		"BERNARD_APP_REGION",
	}
)

type RegionConfig struct {
	AuthRegion         string
	ResourceManagerAPI string
	ByteQuotaAPI       string
	JwtAuthUrlPrefix   string
	ScheduledMatchFE   string
}

func getAppRegion() AppRegion {
	appRegion := AppRegionProdCN
	for _, env := range AppRegionEnvs {
		if region := os.Getenv(env); region != "" {
			appRegion = AppRegion(region)
			break
		}
	}
	return appRegion
}

func GetRegionConfig() *RegionConfig {
	appRegion := getAppRegion()
	switch appRegion {
	case AppRegionProdCN:
		return &RegionConfig{
			ResourceManagerAPI: CNResourceManagerAPI,
			AuthRegion:         CNAuthRegion,
			ByteQuotaAPI:       CNByteQuotaAPI,
			JwtAuthUrlPrefix:   CNJwtAuthUrlPrefix,
			ScheduledMatchFE:   CNScheduledMatchFE,
		}
	case AppRegionProdUS:
		return &RegionConfig{
			AuthRegion:       ProdUSAuthRegion,
			JwtAuthUrlPrefix: ProdUSJwtAuthUrlPrefix,
			ScheduledMatchFE: ProdUSScheduledMatchFE,
		}
	case AppRegionProdI18NBD:
		return &RegionConfig{
			AuthRegion:       ProdI18NBDAuthRegion,
			JwtAuthUrlPrefix: ProdI18NBDJwtAuthUrlPrefix,
		}
	case AppRegionProdI18N:
		return &RegionConfig{
			ResourceManagerAPI: ProdI18NResourceManagerAPI,
			AuthRegion:         ProdI18NAuthRegion,
			ByteQuotaAPI:       ProdI18NByteQuotaAPI,
			JwtAuthUrlPrefix:   ProdI18NJwtAuthUrlPrefix,
			ScheduledMatchFE:   ProdI18NScheduledMatchFE,
		}
	case AppRegionProdTTP, AppRegionTTP:
		return &RegionConfig{
			ResourceManagerAPI: TTPUSResourceManagerAPI,
			ByteQuotaAPI:       TTPUSByteQuotaAPI,
			JwtAuthUrlPrefix:   TTPUSJwtAuthUrlPrefix,
			ScheduledMatchFE:   TTPUScheduledMatchFE,
		}
	case AppRegionBoe:
		return &RegionConfig{
			ResourceManagerAPI: BoeResourceManagerAPI,
			JwtAuthUrlPrefix:   BoeJwtAuthUrlPrefix,
		}
	case AppRegionTesting, AppRegionSandbox:
		return &RegionConfig{
			ResourceManagerAPI: TestingResourceManagerAPI,
			AuthRegion:         TestingAuthRegion,
			ByteQuotaAPI:       TestingByteQuotaAPI,
			JwtAuthUrlPrefix:   TestingJwtAuthUrlPrefix,
			ScheduledMatchFE:   TestingScheduledMatchFE,
		}
	default:
		return nil
	}
}
