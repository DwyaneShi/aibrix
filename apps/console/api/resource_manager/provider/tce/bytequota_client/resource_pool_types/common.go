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

package resource_pool_types

import (
	"time"
)

const (
	KindResourceGroup                                      = "resource-groups"
	KindResourcePool                                       = "resource-pools"
	KindSystemMeta                                         = "system-meta"
	KindResourceReservation                                = "resource-reservation"
	KindWorkload                                           = "workloads"
	KindLockingItem                                        = "locking-items"
	KindSharingScope                                       = "sharing-scopes"
	KindChangelog                                          = "changelog"
	KindIDSpace                                            = "id-spaces"
	KindInstancePackage                                    = "instance-packages"
	KindElasticScenes                                      = "elastic-scenes"
	KindSupplyDomain                                       = "supply-domains"
	KindTicket                                             = "tickets"
	KindAdmin                                              = "admins"
	KindDynamicConfig                                      = "dynamic-config"
	KindBusinessLine                                       = "business-lines"
	KindCRSObject                                          = "crs-object"
	KindScheduledQuotaRecord                               = "scheduled-quota-record"
	KindResourceScheduledQuota                             = "resource-scheduled-quota"
	KindResourceScheduledQuotaDeletion                     = "resource-scheduled-quota-deletion"
	KindResourceScheduledQuotaDeletionDetermineSkipCluster = "resource-scheduled-quota-deletion-determine-skip-cluster"
	KindScheduledQuotaChangelog                            = "scheduled-quota-changelog"
	KindScheduledResourcePool                              = "scheduled-resource-pool"
	KindResourceLock                                       = "resource-lock"
	KindClusterRecord                                      = "cluster-record"
	KindNodeInfos                                          = "node-infos"
	KindTopologyS1                                         = "topology-s1"
	KindTopologyS2                                         = "topology-s2"

	// do not include special characters (like "-") in platform string
	PlatformTCE          = "Compute"
	PlatformTCEByteDrive = "BytedriveTCE"
	PlatformIAAS         = "IAAS"
	PlatformBKE          = "BKE"
	PlatformRedis        = "redis"
	PlatformRDS          = "rds"
	PlatformByteDoc      = "bytedoc"
	PlatformAbase2       = "abase2"
	PlatformByteES       = "bytees"
	PlatformByteKV       = "bytekv"
	PlatformHDFS         = "hdfs"
	PlatformHBase        = "hbase"
	PlatformTOS          = "tos"
	PlatformBMQ          = "bmq"
	PlatformRocketMQ     = "rocketmq"
	PlatformArk          = "ark"

	// labels & annotations
	DC                   = "dc"
	AZ                   = "az"
	PhysicalCluster      = "physical_cluster"
	LogicalCluster       = "logical_cluster"
	Namespace            = "namespace"
	Zone                 = "zone"
	QueueName            = "queue_name"
	ResourceLevel        = "resource_level"
	Env                  = "env"
	Type                 = "type"
	AccountId            = "account_id"
	ByteDriveCluster     = "bytedrive_cluster"
	ByteDriveVolumeType  = "bytedrive_volume_type"
	Offline              = "offline"
	Online               = "online"
	Megatron             = "megatron"
	Norbert              = "norbert"
	ServerlessType       = "serverless"
	Guarantee            = "guarantee"
	BestEffort           = "besteffort"
	NodeLevel            = "nodeLevel"
	TCENormalQueueName   = "default"
	TCEIncQueueName      = "inc"
	CPU                  = "cpu"
	Memory               = "memory"
	Socket               = "socket"
	PKG                  = "package"
	FlexSocket           = "flex-socket"
	CPUOptimized         = "cpu-optimized"
	CPUOnDemand          = "cpu-on-demand"
	CPUOptimizedOnDemand = "cpu-optimized-on-demand"
	BestEffortCPU        = "best-effort-cpu"
	Default              = "default"
	//config name
	DynamicConfigName = "conf"

	ClusterPartition = "partition"
	// Partition is not need by single cluster, so set "" as a placeholder
	SingleClusterPartition = ""

	Gallipoli     = "Gallipoli"
	BernardProd   = "Bernard-Prod"
	Dolores       = "Dolores"
	Echo          = "Echo"
	CloudNative   = "CloudNative"
	Seed01        = "Seed01"
	Seed02        = "Seed02"
	ResoptSandbox = "ResOptSandbox"
	Fedtest       = "Fedtest"

	FederationMember = "federationMember"
	StockCRS         = "quota.bytedance.com/uce"

	//operator
	OperatorNotFromQuota = "OperatorNotFromQuota"

	StableGuarantee      = "stableGuarantee"
	FluctuatingGuarantee = "fluctuatingGuarantee"

	// extended guarantee resource, used in UCE
	// ref: https://bytedance.larkoffice.com/docx/YOlXdoG5moQtTCxeqGEc4zJYnYg
	ExtendedGuaranteeBuffer     = "extendedGuaranteeBuffer"
	ExtendedGuaranteeOndemandV2 = "extendedGuaranteeOndemandV2"
	ExtendedGuaranteeSpot       = "extendedGuaranteeSpot"

	// permission types quota support
	PersonAccount  = "person_account"
	ServiceAccount = "service_account"

	TCE      = "tce"
	Bernard  = "bernard"
	CNCore   = "核"
	BPM      = "bpm"
	API      = "api"
	Expand   = "expand"
	Reduce   = "reduce"
	Quota    = "quota"
	Finished = "finished"
	Rejected = "rejected"
	// language
	Zh    = "zh"
	En    = "en"
	Zh_En = "zh_en"

	//gray
	KindGrayK8sCluster = "gray_k8s_cluster"

	// ondemand hours
	FirstHour  = 0
	LastHour   = 23
	HoursInDay = 24

	// qos
	Dedicated      = "dedicated"
	Shared         = "shared"
	Reclaimed      = "reclaimed"
	Any            = "any"
	DedicatedCores = "dedicated_cores"

	// sale mode
	Reserved  = "reserved"
	Ondemand  = "ondemand"
	Spot      = "spot"
	Scheduled = "scheduled"

	// approval rule type
	GeneralApprovalRuleType = "general"

	SyncStatusPending  = "pending"
	SyncStatusFinished = "finished"

	ArkForTTPoolType = "tark"
	AiPaasPoolType   = "aipaas"

	RedisKeyPSMAllParentsNodeIDListMap = "psm_all_parents_node_id_list_map"
)

var (
	ValidPlatforms = []string{
		PlatformTCE, PlatformTCEByteDrive, PlatformIAAS, PlatformBKE,
		PlatformRedis, PlatformRDS, PlatformByteDoc, PlatformAbase2,
		PlatformByteES, PlatformByteKV, PlatformHDFS, PlatformHBase,
		PlatformTOS, PlatformBMQ, PlatformRocketMQ, PlatformArk,
	}
	ValidStoragePlatforms = []string{
		PlatformRedis, PlatformRDS, PlatformByteDoc, PlatformAbase2,
		PlatformByteES, PlatformByteKV, PlatformHDFS, PlatformHBase,
		PlatformTOS, PlatformBMQ, PlatformRocketMQ, PlatformArk,
	}
	UserDefinePackages = map[string]map[string]int64{
		"Juliett/default": {
			"c2.1x":  1000000,
			"c2.2x":  1000000,
			"c3.1x":  1000000,
			"c3.2x":  1000000,
			"c3.4x":  1000000,
			"c2.1xn": 1000000,
			"c2.2xn": 1000000,
			"c2.4xn": 1000000,
			"g2.2x":  1000000,
			"g2.1x":  1000000,
			"g3.1xn": 1000000,
		},
		"Juliett/yarn": {
			"c2.1x":  1000000,
			"c2.2x":  1000000,
			"c3.1x":  1000000,
			"c3.2x":  1000000,
			"c3.4x":  1000000,
			"c2.1xn": 1000000,
			"c2.2xn": 1000000,
			"c2.4xn": 1000000,
			"g2.2x":  1000000,
			"g2.1x":  1000000,
			"g3.1xn": 1000000,
		},
		/*
			"Juliett/ai": {
				"ae2.1x-nvidia.t4.16g.2x": 10000,
				"ae2.2x-nvidia.t4.16g.4x": 10000,
				"ae2.4x-nvidia.t4.16g.8x": 10000,
				"ae4.1x-nvidia.t4.16g.4x": 10000,
				"ae4.2x-nvidia.t4.16g.8x": 10000,
				"a2.1x-nvidia.t4.16g.1x":  10000,
				"a2.2x-nvidia.t4.16g.2x":  10000,
				"a2.4x-nvidia.t4.16g.4x":  10000,
				"a4.2x-nvidia.t4.16g.4x":  10000,
				"a4.2xs-nvidia.t4.16g.4x": 10000,
			},
		*/
	}
	OnDemandV1TceClusters  = []string{}
	OnDemandV1TceWorkHours = []int{19, 20, 21, 22, 23}

	OnDemandV1FaasClusters = []string{
		// FaaS Microservice Cluster
		"FaaSSandBox/default",
		"Faas-boe/bytefaas",
		"BOE-i18n/bytefaas",
		"Ginger/bytefaas",
		"Faas-lf/default",
		"Faas/default",
		"Caijing/bytefaas",
		"Olearia/bytefaas",
		"Payments/bytefaas",
		"Zeil/bytefaas",
		"Kaiser/bytefaas",
		"Normal01/bytefaas",
		"FaaS01/default",

		// FaaS Socket Cluster
		"FaaSSandBox/bytefaas-socket",
		"Faas/bytefaas-socket",
		"Faas-lf/bytefaas-socket",
	}
	OnDemandV1FaasWorkHours = []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

	OnDemandV2Clusters = []string{
		//"ResOptSandbox/default",
		//"solarng/default",
		"federation/default/lf",
		"fedtest/default/lf",
		"fedtest/default/boe",
		"koala/default/yg",
		"koala/default/lf",
		"koala/default/hl",
		"koala/default/lq",
		"koala/default/gl",
		"koala/amd/yg",
		"koala/amd/lf",
		"koala/amd/hl",
		"koala/amd/lq",
		"llama/default/lf",
		"llama/hdfstest/lf",
		"zebra/ubiquity/hl",
		"candy/lagrange-stable-cpu/hl",
		"candy/lagrange-stable-worker-cpu/hl",
		"candy/dandelion-ai-g12/hl",
		"juliett/topology/hl",
		"juliett/topology/lq",
		"juliett/dandelion-ai-g12/hl",
		"juliett/dandelion-ai-g12/lq",
		"lambs/default/lf",
		"juliett/yarn/lf",
		"juliett/yarn/hl",
		"juliett/yarn/lq",
		"juliett/yarn/yg",
		"juliett/default/lf",
		"juliett/default/hl",
		"juliett/default/lq",
		"juliett/default/yg",
		"juliett/topology-dandelion/yg",
		"juliett/ai/hl",
		"juliett/ai/lq",
		"juliett/dandelion-ai-g19/hl",
		"juliett/dandelion-ai-g19/yg",
		"rabbit/default/hl",
		"rabbit/default/lf",
		"rabbit/amd/hl",
		"bear/default/hl",
		"bear/default/yg",
		"bear/default/gl",
		"bear/amd/hl",
		"panda/default/lf",
		"panda/default/hl",
		"panda/default/lq",
		"panda/amd/lf",
		"panda/amd/hl",
		"panda/amd/lq",
		"zelda/ubiquity/lq",
		"zelda/ubiquity/yg",
		"feins/default/hl",
		"feins/flink/hl",
		"feins/flink-amd/hl",
		"feins/flink-hybrid-hdd/hl",
		"feins/flink-hybrid-ssd/hl",
		"feins/default/lf",
		"feins/flink/lf",
		"feins/flink-amd/lf",
		"feins/flink-hybrid-hdd/lf",
		"feins/flink-hybrid-ssd/lf",
		"feins/default/lq",
		"feins/flink/lq",
		"feins/flink-amd/lq",
		"feins/flink-hybrid-hdd/lq",
		"feins/flink-hybrid-ssd/lq",
		"florin/default/hl",
		"florin/flink/hl",
	}

	AnyQosZones = []string{
		"China-North",
		"China-North-LF",
		"China-East",
	}

	TarkQueueTypes = []string{
		"ep",
		"batch",
		"mcj",
		"ptu",
	}
)

var AllowedFedPartitions = []string{
	MicroPartition, SocketPartition, GPUPartition, YodelPartition,
}

type RecordMeta struct {
	CreatedAt time.Time `json:"created_at" bson:"created_at"`
	UpdatedAt time.Time `json:"updated_at" bson:"updated_at"`
	IsDeleted bool      `json:"is_deleted" bson:"is_deleted"`
}

type Page struct {
	Pn    int `json:"pn"`
	Rn    int `json:"rn"`
	Total int `json:"total"`
}

type Topic string
