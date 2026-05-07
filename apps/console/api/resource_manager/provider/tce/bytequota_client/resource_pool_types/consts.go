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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// object kind
const (
	EmptyKind = "" // usually used by vk pod

	PodKind                  = "Pod"
	ReplicaSetKind           = "ReplicaSet"
	DeploymentKind           = "Deployment"
	StormServiceKind         = "StormService"
	RoleSetKind              = "RoleSet"
	StatefulSetExtensionKind = "StatefulSetExtension"
	SolarServiceKind         = "SolarService"
	FedDeploymentKind        = "FederatedDeployment"

	ArkApplicationWorkloadKind = "ArkApplicationWorkload"
	ApplicationKind            = "Application"
	WorkerSetKind              = "WorkerSet"
	WorkerKind                 = "Worker"
	MLJobKind                  = "MLJob"
	MLJobWorkloadKind          = "MLJobWorkload"

	HPAKind                 = "HorizontalPodAutoScaleExtension"
	FedHPAKind              = "FedHPA"
	ScaleGroupKind          = "ScaleGroup"
	DefaultHPAReferenceName = "*"

	DeploymentWorkloadKind   = "DeploymentWorkload"
	SolarServiceWorkloadKind = "SolarServiceWorkload"
	StormServiceWorkloadKind = "StormServiceWorkload"
	HPATreatmentKind         = "HPATreatment"
	ScaleGroupTreatmentKind  = "ScaleGroupTreatment"

	RayClusterKind = "RayCluster"

	YarnApplicationKind = "Application"

	VcJobKind       = "Job"
	TrainingJobKind = "TrainingJob"
	EurusTrialKind  = "Trial"

	PrimusRoleKind = "PrimusRole"
	PrimusJobKind  = "PrimusJob"
	NorbertJobKind = "NorbertJob"

	ArceePodSetKind      = "ArceePodSet"
	ArceeApplicationKind = "ArceeApplication"
	WorkspaceWorkerKind  = "Worker"
	WorkspaceKind        = "Workspace"

	QueueResourceReservationKind = "QueueResourceReservation"

	MicroPartition     = "micro"
	SocketPartition    = "socket"
	MicroGPUPartition  = "microgpu"
	SocketGPUPartition = "socketgpu"
	YodelPartition     = "yodel"
	GPUPartition       = "gpu"
	EmptyPartition     = ""
)

// annotations
const (
	// keys

	// daemon
	DaemonDeploymentAnnotationKey = "deployment.kubernetes.io/daemon-deployment"
	DaemonPodAnnotationKey        = "pod.tce.kubernetes.io/tce-daemon"
	DaemonWorkloadAnnotationKey   = "workload.octagram.io/daemon"

	// quota
	AnnoKeyNamespaceType              = "namespace_type"
	AnnoKeyTopologyAffinityRestricted = "topology_affinity_restricted"

	AnnoValueAntiHostNs                     = "anti_host_ns"
	AnnoValueTopologyAffinityRestrictedTrue = "true"

	LastUpdateTimeAnnotationKey = "lastUpdate"

	GPUTypeAnnotationKey     = "deployment.tce.kubernetes.io/gpu-type"
	HabanaTypeAnnotationKey  = "deployment.tce.kubernetes.io/habana-type"
	XPUTypeAnnotationKey     = "deployment.tce.kubernetes.io/xpu-type"
	NPUTypeAnnotationKey     = "deployment.tce.kubernetes.io/npu-type"
	PackageTypeAnnotationKey = "instance-model"

	DisplayQueueNameAnnotationKey       = "queues.tce.byted/queueDisplayName"
	GodelQueueNameAnnotationKey         = "godel.bytedance.com/queue-name"
	QueueNameAnnotationKey              = "queue-name"
	GodelReservationNameAnnotationKey   = "godel.bytedance.com/reservation-name"
	ReservationNameAnnotationKey        = "reservation-name"
	IgnoreQuotaAnnotationKey            = "tce.kubernetes.io/ignore-quota"
	QueueResourcePoolTypeAnnotationKey  = "queues.tce.byted/resourcePoolType"
	QueueResourcePoolNameAnnotationKey  = "queues.tce.byted/resourcePoolName"
	PodResourceTypeAnnotationKey        = "godel.bytedance.com/pod-resource-type"
	QueueQoSLevelAnnotationKey          = "godel.bytedance.com/queue_qos_level"
	PodQoSLevelAnnotationKey            = "katalyst.cloud/qos_level"
	KatalystPodQoSLevelAnnotationKey    = "katalyst.kubewharf.io/qos_level"
	PodTypeAnnotationKey                = "godel.bytedance.com/pod-type"
	PodSaleModeAnnotationKey            = "bytedance.quota.salemode"
	ScheduledToTerminateAtAnnotationKey = "godel.bytedance.com/scheduled-to-terminate-at"
	CountQuotaUsageByWorkload           = "godel.bytedance.com/count-quota-usage-by-workload"

	KatalystPodMemoryEnahancementAnnotationKey = "katalyst.kubewharf.io/memory_enhancement"
	KatalystMemoryNumaBindingAnnotationKey     = "numa_binding"

	QuotaAccountingPolicyAnnotationKey = "godel.bytedance.com/queue-accounting-policy" // ns or queue annotation
	UserSetResourcesAnnotationKey      = "godel.bytedance.com/user-set-resources"      // pod annotation
	QuotaOverrideAnnotationKey         = "godel.bytedance.com/quota-overrides"

	// quota queue resource reservation
	ReserveToAutoScale        = "auto_scale"
	ReserveToScheduledLending = "scheduled_lending"

	// value
	PodResourceTypeBestEffort = "best-effort"
	PodResourceTypeGuaranteed = "guaranteed"

	// scale group
	ScaleGroupReferAnnotationKey = "pod.tce.kubernetes.io/scale-group-referer-type"
	ScaleGroupReferSolarService  = "solarservice"
	ScaleGroupReferDeployment    = "deployment"

	// lifecycle
	LifeCycleReadyAnnotationKey    = "lifecycle.tce.byted.org/Ready"
	LifeCycleRecordedAnnotationKey = "lifecycle.tce.byted.org/Recorded"

	CanaryDeploymentAnnotationKey = "deployment.tce.kubernetes.io/canary-deployment"

	DeploymentIgnoreNotReadyAnnotationKey    = "scheduler.alpha.kubernetes.io/daemon-include-not-ready-node"
	DeploymentUseHybridResourceAnnotationKey = "deployment.kubernetes.io/use-hybrid-resource"

	WorkloadQoSLevelFeatureGateKey     = "KatalystQosLevel"
	WorkloadSpecifiedTimeAnnotationKey = "bytedance.quota.timespecified"

	FedBatchQueueManagedAnnotationKey = "queues.tce.byted/fedBatchQueueManaged"

	NodeScheduledAnnotationKey   = "node.tce.byted.org/scheduled-lending"
	NodeReservationAnnotationKey = "node.tce.byted.org/reservation"

	FederationExpectedHPATreatmentRuleAnnotationKey = "kubeadmiral.io/expected-hpa-treatment-rules"
)

// labels
const (
	// key

	AcceleratorLabelKey      = "accelerator"
	ShadowDeployLabelKey     = "requireFastRecover"
	NodeLevelLabelKey        = "nodeLevel"
	VirtualNodeLevelLabelKey = "virtualNodeLevel"
	ScenarioLabelKey         = "scenario"
	NodeNameLabelKey         = "nodeName"

	NodeReservationLabelKey = "node.tce.byted.org/reserved"

	PartitionLabelKey        = "partition"
	FederationMemberLabelKey = "federationMember"

	DCLabelKey                  = "dc"
	ZoneLabelKey                = "zone"
	ResourceLevelLabelKey       = "resource_level"
	ClusterNameLabelKey         = "physical_cluster"
	PhysicalClusterNameLabelKey = "physical-cluster-name"

	// value

	VKNodeTypeLabelValue     = "virtual-kubelet"
	ScenarioScheduledLending = "scheduled-lending"
)

// taint
const (
	IPv6OnlyTaintKey    = "IPv6Only"
	ScheduledTaintKey   = "NoScheduleByHybridController"
	ScheduledTaintToD   = "tod"
	ReservationTaintKey = "NoScheduleByReservation"
	SeedLendingTaintKey = "ResourceLending"
)

// resources
const (
	ResourceCPU              v1.ResourceName = "cpu"
	ResourceMemory           v1.ResourceName = "memory"
	ResourceSocket           v1.ResourceName = "bytedance.com/socket"
	ResourceHabana           v1.ResourceName = "habana.ai/goya"
	ResourceCodec            v1.ResourceName = "nvidia.com/codec"
	ResourceShareGPU         v1.ResourceName = "nvidia.com/share-gpu"
	ResourceGPU              v1.ResourceName = "nvidia.com/gpu"
	ResourceEXGPU            v1.ResourceName = "nvidia.com/ex-gpu"
	ResourceBEGPU            v1.ResourceName = "nvidia.com/be-gpu"
	ResourceNPU              v1.ResourceName = "hw.com/npu"
	ResourceXPU              v1.ResourceName = "bytedance.com/xpu"
	ResourceExCPU            v1.ResourceName = "cnr.godel.org/ex-cpu"
	ResourceExMemory         v1.ResourceName = "cnr.godel.org/ex-memory"
	ResourceExSocket         v1.ResourceName = "cnr.godel.org/ex-socket"
	ResourceExGPU            v1.ResourceName = "nvidia.com/ex-gpu"
	ResourceNic              v1.ResourceName = "bytedance.com/nic"
	ResourceNs2Bandwidth     v1.ResourceName = "bytedance.com/ns2-bandwidth" //未直接上报，KCNR转换过来的
	ResourceKCNRNetBandwidth v1.ResourceName = "resource.katalyst.kubewharf.io/net_bandwidth"
)

// kcnr related
const (
	KCNRChildTypeNIC = "NIC"

	KCNRAttributeNameNetNsName = "katalyst.kubewharf.io/netns_name"

	KCNRAttributeValueNetNs2 = "ns2"
)

type QoSLevel string
type SaleModeType string
type PodResourceType string
type ExtensionType string

const (
	PodQoSLevelDedicated = QoSLevel("dedicated_cores")
	PodQoSLevelShared    = QoSLevel("shared_cores")
	PodQoSLevelReclaimed = QoSLevel("reclaimed_cores")
	PodQoSLevelAny       = QoSLevel("any")

	SaleModeReserved = SaleModeType("reserved")
	// SaleModeReservedNew tags the data from reservedStat fields
	// TODO: delete it after we transfer from guaranteedXXX to reservedStat
	SaleModeReservedNew = SaleModeType("reserved_new")
	SaleModeScheduled   = SaleModeType("scheduled")
	SaleModeOnDemand    = SaleModeType("on-demand")
	SaleModeSpot        = SaleModeType("spot")

	ExtensionTopology = ExtensionType("topology")

	GuaranteeResource  = "Guaranteed"
	BestEffortResource = "BestEffort"
	TideResource       = "Tide"
)

// enum
type QuotaResourceName = string

const (
	QuotaResourceNameCPU      QuotaResourceName = "cpu"
	QuotaResourceNameMemory   QuotaResourceName = "memory"
	QuotaResourceNameGPU      QuotaResourceName = "gpu"
	QuotaResourceNameSocket   QuotaResourceName = "socket"
	QuotaResourceNameXPU      QuotaResourceName = "xpu"
	QuotaResourceNameNPU      QuotaResourceName = "npu"
	QuotaResourceNameHabana   QuotaResourceName = "habana"
	QuotaResourceNameCodec    QuotaResourceName = "codec"
	QuotaResourceNameNIC      QuotaResourceName = "nic" //network interface card
	QuotaResourceNameNbw      QuotaResourceName = "nbw" //network bandwidth
	QuotaResourceNamePackage  QuotaResourceName = "instanceModel"
	SupplyResourceNamePackage QuotaResourceName = "package"
)

type QuotaResourceType = string

const (
	DefaultQuotaResourceType QuotaResourceType = "default"
)

var (
	ResourceQuotaNameMap = map[v1.ResourceName]QuotaResourceName{
		ResourceSocket:           QuotaResourceNameSocket,
		ResourceGPU:              QuotaResourceNameGPU,
		ResourceBEGPU:            QuotaResourceNameGPU,
		ResourceEXGPU:            GPUTypeAnnotationKey, // TODO: TBD
		ResourceShareGPU:         QuotaResourceNameGPU,
		ResourceHabana:           QuotaResourceNameHabana,
		ResourceCodec:            QuotaResourceNameCodec,
		ResourceXPU:              QuotaResourceNameXPU,
		ResourceNPU:              QuotaResourceNameNPU,
		ResourceNic:              QuotaResourceNameNIC,
		ResourceNs2Bandwidth:     QuotaResourceNameNbw,
		ResourceKCNRNetBandwidth: QuotaResourceNameNbw,
	}
	ResourceAnnotationTypeKeyMap = map[v1.ResourceName]string{
		ResourceSocket:           PackageTypeAnnotationKey,
		ResourceGPU:              GPUTypeAnnotationKey,
		ResourceBEGPU:            GPUTypeAnnotationKey,
		ResourceShareGPU:         GPUTypeAnnotationKey,
		ResourceHabana:           HabanaTypeAnnotationKey,
		ResourceCodec:            GPUTypeAnnotationKey,
		ResourceXPU:              XPUTypeAnnotationKey,
		ResourceNPU:              NPUTypeAnnotationKey,
		ResourceKCNRNetBandwidth: QuotaResourceNameNbw,
	}
	ResourceNodeSelectorTypeKeyMap = map[v1.ResourceName]string{
		ResourceGPU:      AcceleratorLabelKey,
		ResourceBEGPU:    AcceleratorLabelKey,
		ResourceShareGPU: AcceleratorLabelKey,
		ResourceHabana:   AcceleratorLabelKey,
		ResourceCodec:    AcceleratorLabelKey,
		ResourceXPU:      AcceleratorLabelKey,
		ResourceNPU:      AcceleratorLabelKey,
	}

	TypeSensitiveResources = []QuotaResourceName{
		QuotaResourceNameGPU,
		QuotaResourceNameNPU,
		QuotaResourceNameXPU,
		QuotaResourceName(ResourceEXGPU),
	}
)

var (
	ResourceCheckList = map[QuotaResourceName]struct{}{
		QuotaResourceNameCPU:    {},
		QuotaResourceNameMemory: {},
		QuotaResourceNameSocket: {},
		QuotaResourceNameGPU:    {},
		QuotaResourceNameHabana: {},
		QuotaResourceNameXPU:    {},
		QuotaResourceNameNPU:    {},
		QuotaResourceNameNIC:    {},
		QuotaResourceNameNbw:    {},
	}

	// topo quota ignore cpu and memory to make crs cleaner
	TopoResourceRecordList = map[QuotaResourceName]struct{}{
		QuotaResourceNameGPU:      {},
		QuotaResourceNameXPU:      {},
		QuotaResourceNameNPU:      {},
		QuotaResourceNamePackage:  {},
		SupplyResourceNamePackage: {},
	}

	SocketRequiredResources = map[string]struct{}{
		QuotaResourceNameCPU:    {},
		QuotaResourceNameMemory: {},
		QuotaResourceNameXPU:    {},
		QuotaResourceNameGPU:    {},
		QuotaResourceNameNPU:    {},
	}

	// NodeToQueueResourceKey key: resource key used by node, value: resource key used by clusterresourcessupply/Queue
	NodeToQueueResourceKey = map[v1.ResourceName]QuotaResourceName{
		ResourceCPU:              QuotaResourceNameCPU,
		ResourceMemory:           QuotaResourceNameMemory,
		ResourceSocket:           QuotaResourceNameSocket,
		ResourceGPU:              QuotaResourceNameGPU,
		ResourceShareGPU:         QuotaResourceNameGPU,
		ResourceHabana:           QuotaResourceNameHabana,
		ResourceCodec:            QuotaResourceNameCodec,
		ResourceNPU:              QuotaResourceNameNPU,
		ResourceXPU:              QuotaResourceNameXPU,
		ResourceNic:              QuotaResourceNameNIC,
		ResourceKCNRNetBandwidth: QuotaResourceNameNbw,
	}

	// CNRToQueueResourceKey key: resource key used by cnr, value: resource key used by clusterresourcessupply/Queue
	CNRToQueueResourceKey = map[v1.ResourceName]QuotaResourceName{
		// gt/shared-be resources supported by physical(vk backend)/vk frontend clusters
		ResourceCPU:          QuotaResourceNameCPU,
		ResourceMemory:       QuotaResourceNameMemory,
		ResourceGPU:          QuotaResourceNameGPU,
		ResourceNPU:          QuotaResourceNameNPU,
		ResourceXPU:          QuotaResourceNameXPU,
		ResourceNs2Bandwidth: QuotaResourceNameNbw,

		// exclusive-be resources names supported by vk frontend clusters
		ResourceExCPU:    QuotaResourceName(ResourceExCPU),
		ResourceExMemory: QuotaResourceName(ResourceExMemory),
		ResourceExSocket: QuotaResourceName(ResourceExSocket),
		ResourceExGPU:    QuotaResourceName(ResourceExGPU),
	}

	KCNRToQueueResourceKey = map[v1.ResourceName]QuotaResourceName{
		ResourceNs2Bandwidth: QuotaResourceNameNbw,
	}
)

// resource list type
const (
	UnknownResourceType string = "unknown"
)

var (
	TimeZoneMap = map[string]string{
		"cn":  "Asia/Shanghai",
		"utc": "UTC",
		"va":  "America/New_York",
		"sg":  "Asia/Shanghai", // the timezone in singapore has changed for historical reason, from 1982 till now (2019), it will be same as Asia/Shanghai
	}
)

var (
	PackageResourceFactor = map[v1.ResourceName]float64{
		v1.ResourceCPU:    2.0,
		v1.ResourceMemory: 2.0,
	}
)

// cluster feature-gates
const (
	LogicalClusterLPVLVMSupported         = "lpv-lvm"
	LogicalClusterLPVBlockDeviceSupported = "lpv-block_device"
)

const (
	PoolTypeDedicatedCores   = "dedicated"
	PoolTypeShareNumaBinding = "share_numa_binding"
	PoolTypeSharedCores      = "share"
)

const DefaultResourcePoolNamespace = "default"

var (
	BernardClusters = map[string]bool{
		"bernard":      true,
		"bernard-prod": true,
		"gallipoli":    true,
		"dolores":      true,
		"echo":         true,
		"seed01":       true,
		"seed02":       true,
	}
)

var (
	FedHPAQualifiedResource     = schema.GroupResource{Group: "types.federated.tce.byted.org", Resource: "federatedHorizontalPodAutoScaleExtension"}
	FedeployQualifiedResource   = schema.GroupResource{Group: "types.federated.tce.byted.org", Resource: "FederatedDeployment"}
	EurusTrialQualifiedResource = schema.GroupResource{Group: "eurus.bytedance.com", Resource: "Trial"}
	PrimusJobQualifiedResource  = schema.GroupResource{Group: "norbert.k8s.io", Resource: "PrimusJob"}
)

const (
	TCEPlatform      = "compute"
	MegatronPlatform = "megatron"
	VK               = "vk"
)

// Federation
var FederationQueuePrefixs = []string{"federation", "fedtest"}

// object fields name

const (
	QueueCapacity  = "Capacity"
	QueueAllocated = "Allocated"

	GuaranteeCapacity    = "GuaranteedCapacity"
	GuaranteeAllocatable = "GuaranteeAllocatable"
	BestEffortCapacity   = "BestEffortCapacity"
	TideAllocatable      = "TideAllocatable"
	TideSupply           = "TideSupply"
)

const (
	ValueTrue  = "true"
	ValueFalse = "false"
)

type Scene string

const (
	ToB Scene = "ToB"
	ToD Scene = "ToD"
)

const (
	DirectionToB2ToD = "ToB2ToD"
	DirectionToD2ToB = "ToD2ToB"
)

const (
	ScenarioKey      = "scenario"
	QueueReservation = "QueueReservation"
	ScheduledLending = "scheduled-lending"
)

// const for godel reservation
// https://code.byted.org/godel/godel-deployment/blob/master/crds/scheduling.orchestration.bytedance.com_reservations.yaml
const (
	ReservationPlaceHolderPodAnnotation = "godel.bytedance.com/reservation-placeholder"
	ReservationIndexAnnotation          = "godel.bytedance.com/reservation-index"
	PlaceholderPodUIDAnno               = "godel.bytedance.com/placeholder-uid"
	ReservationOwnerTypeAnno            = "godel.bytedance.com/reservation-owner-type"
	ReservationOwnerNameAnno            = "godel.bytedance.com/reservation-owner"
	ReservationOriginalPodNameAnno      = "godel.bytedance.com/reservation-original-pod"

	ReservationPlaceholderPostFix = "-placeholder"
)

// GodelReservationLegalAnnotationKeys are the legal annotation keys for godel reservation
var GodelReservationLegalAnnotationKeys = []string{
	ReservationIndexAnnotation,
	PlaceholderPodUIDAnno,
	ReservationOriginalPodNameAnno,
	PackageTypeAnnotationKey,
	ReservationOwnerTypeAnno,
	ReservationOwnerNameAnno,
}
