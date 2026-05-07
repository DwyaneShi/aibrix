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

package scheduled_plan_types

import (
	"time"
)

// Defines values for AcceleratorPreferencePrecisionSupportPreferred.
const (
	AcceleratorPreferencePrecisionSupportPreferredBF16 AcceleratorPreferencePrecisionSupportPreferred = "BF16"
	AcceleratorPreferencePrecisionSupportPreferredFP16 AcceleratorPreferencePrecisionSupportPreferred = "FP16"
	AcceleratorPreferencePrecisionSupportPreferredFP32 AcceleratorPreferencePrecisionSupportPreferred = "FP32"
	AcceleratorPreferencePrecisionSupportPreferredFP4  AcceleratorPreferencePrecisionSupportPreferred = "FP4"
	AcceleratorPreferencePrecisionSupportPreferredFP64 AcceleratorPreferencePrecisionSupportPreferred = "FP64"
	AcceleratorPreferencePrecisionSupportPreferredFP8  AcceleratorPreferencePrecisionSupportPreferred = "FP8"
	AcceleratorPreferencePrecisionSupportPreferredINT4 AcceleratorPreferencePrecisionSupportPreferred = "INT4"
	AcceleratorPreferencePrecisionSupportPreferredINT8 AcceleratorPreferencePrecisionSupportPreferred = "INT8"
	AcceleratorPreferencePrecisionSupportPreferredTF32 AcceleratorPreferencePrecisionSupportPreferred = "TF32"
)

// Defines values for AcceleratorPreferencePrecisionSupportRequired.
const (
	AcceleratorPreferencePrecisionSupportRequiredBF16 AcceleratorPreferencePrecisionSupportRequired = "BF16"
	AcceleratorPreferencePrecisionSupportRequiredFP16 AcceleratorPreferencePrecisionSupportRequired = "FP16"
	AcceleratorPreferencePrecisionSupportRequiredFP32 AcceleratorPreferencePrecisionSupportRequired = "FP32"
	AcceleratorPreferencePrecisionSupportRequiredFP4  AcceleratorPreferencePrecisionSupportRequired = "FP4"
	AcceleratorPreferencePrecisionSupportRequiredFP64 AcceleratorPreferencePrecisionSupportRequired = "FP64"
	AcceleratorPreferencePrecisionSupportRequiredFP8  AcceleratorPreferencePrecisionSupportRequired = "FP8"
	AcceleratorPreferencePrecisionSupportRequiredINT4 AcceleratorPreferencePrecisionSupportRequired = "INT4"
	AcceleratorPreferencePrecisionSupportRequiredINT8 AcceleratorPreferencePrecisionSupportRequired = "INT8"
	AcceleratorPreferencePrecisionSupportRequiredTF32 AcceleratorPreferencePrecisionSupportRequired = "TF32"
)

// Defines values for AffinityPolicyPolicies.
const (
	SameMinipod  AffinityPolicyPolicies = "sameMinipod"
	SameNumaNode AffinityPolicyPolicies = "sameNumaNode"
	SameSwitchS0 AffinityPolicyPolicies = "sameSwitchS0"
	SameSwitchS1 AffinityPolicyPolicies = "sameSwitchS1"
	SameSwitchS2 AffinityPolicyPolicies = "sameSwitchS2"
	SingleHost   AffinityPolicyPolicies = "singleHost"
)

// Defines values for FlexibleAllocationPriority.
const (
	FlexibleAllocationPriorityAny      FlexibleAllocationPriority = "any"
	FlexibleAllocationPriorityEarliest FlexibleAllocationPriority = "earliest"
	FlexibleAllocationPriorityLatest   FlexibleAllocationPriority = "latest"
)

// Defines values for GroupSpecNetworkRdma.
const (
	GroupSpecNetworkRdmaAny        GroupSpecNetworkRdma = "any"
	GroupSpecNetworkRdmaInfiniband GroupSpecNetworkRdma = "infiniband"
	GroupSpecNetworkRdmaIwarp      GroupSpecNetworkRdma = "iwarp"
	GroupSpecNetworkRdmaNone       GroupSpecNetworkRdma = "none"
	GroupSpecNetworkRdmaRoce       GroupSpecNetworkRdma = "roce"
)

// Defines values for MatchingIntentStatus.
const (
	MatchingIntentStatusCanceled            MatchingIntentStatus = "Canceled"
	MatchingIntentStatusCanceling           MatchingIntentStatus = "Canceling"
	MatchingIntentStatusFailed              MatchingIntentStatus = "Failed"
	MatchingIntentStatusFinished            MatchingIntentStatus = "Finished"
	MatchingIntentStatusMatchingBooked      MatchingIntentStatus = "Matching.Booked"
	MatchingIntentStatusMatchingBooking     MatchingIntentStatus = "Matching.Booking"
	MatchingIntentStatusMatchingCommitting  MatchingIntentStatus = "Matching.Committing"
	MatchingIntentStatusMatchingPending     MatchingIntentStatus = "Matching.Pending"
	MatchingIntentStatusMatchingProvisional MatchingIntentStatus = "Matching.Provisional"
	MatchingIntentStatusSucceed             MatchingIntentStatus = "Succeed"
)

// Defines values for MatchingResultStatus.
const (
	MatchingResultStatusBooked      MatchingResultStatus = "booked"
	MatchingResultStatusBooking     MatchingResultStatus = "booking"
	MatchingResultStatusCancelled   MatchingResultStatus = "cancelled"
	MatchingResultStatusCancelling  MatchingResultStatus = "cancelling"
	MatchingResultStatusCommitting  MatchingResultStatus = "committing"
	MatchingResultStatusFailed      MatchingResultStatus = "failed"
	MatchingResultStatusPartial     MatchingResultStatus = "partial"
	MatchingResultStatusPending     MatchingResultStatus = "pending"
	MatchingResultStatusProvisional MatchingResultStatus = "provisional"
	MatchingResultStatusSuccess     MatchingResultStatus = "success"
)

// Defines values for RequesterPlatform.
const (
	Arnold         RequesterPlatform = "Arnold"
	Bernard        RequesterPlatform = "Bernard"
	EngineServing  RequesterPlatform = "Engine-Serving"
	EngineTraining RequesterPlatform = "Engine-Training"
	VeIaaS         RequesterPlatform = "Ve-IaaS"
	VeMLP          RequesterPlatform = "Ve-MLP"
	VeMaaS         RequesterPlatform = "Ve-MaaS"
	VideoArch      RequesterPlatform = "Video-Arch"
)

// Defines values for WorkloadScene.
const (
	DataProcessing WorkloadScene = "data-processing"
	Serving        WorkloadScene = "serving"
	Training       WorkloadScene = "training"
)

// Defines values for WorkloadScheduleCalendar.
const (
	Everyday WorkloadScheduleCalendar = "everyday"
	Holiday  WorkloadScheduleCalendar = "holiday"
	Weekend  WorkloadScheduleCalendar = "weekend"
	Workday  WorkloadScheduleCalendar = "workday"
)

// Defines values for WorkloadScheduleTimeSegmentsPeakLevel.
const (
	High   WorkloadScheduleTimeSegmentsPeakLevel = "high"
	Low    WorkloadScheduleTimeSegmentsPeakLevel = "low"
	Medium WorkloadScheduleTimeSegmentsPeakLevel = "medium"
	Zero   WorkloadScheduleTimeSegmentsPeakLevel = "zero"
)

// AcceleratorPreference 加速器偏好（软约束），用于表达调度优先选择的加速器型号、特性、能力或权重。
// - 非硬性要求，调度器会尽量满足偏好，无法满足时可降级。
// - 适合表达优先型号、带宽、特性、排序权重等。
// - 仅描述单卡能力/特性，不涉及数量和分布。
//
// Accelerator preference (soft constraint), describes preferred accelerator types, features, capabilities, or weights for scheduling. Scheduler will try to satisfy preferences but may downgrade if not possible. Only describes per-card features, not group-level count/distribution.
//
// 典型字段示例：
//
//	preferredTypes: ["NVIDIA A100", "NVIDIA H100"]
//	preferHighBandwidth: true
//	minMemoryGB: 40
//	weight: 10
type AcceleratorPreference struct {
	// Advanced 高级参数，仅高级用户填写
	Advanced *AcceleratorPreferenceAdvanced `json:"advanced,omitempty"`

	// ComputeEquivalenceFactors 自定义算力等价系数，用于异构资源替换场景。
	// - key: 加速器类型
	// - value: 相对于首选类型(preferredTypes[0])的算力等价系数
	//
	// 例如，若 preferredTypes=["L20", "A30", "L40s"]，则：
	// computeEquivalenceFactors: {
	//   "A30": 0.5,  # 1个L20 = 2个A30
	//   "L40s": 2.0  # 1个L40s = 2个L20
	// }
	//
	// 若不提供，则使用系统默认换算关系。
	ComputeEquivalenceFactors *map[string]float32 `json:"computeEquivalenceFactors,omitempty"`

	// MinBandwidthGBps 优先选择显存带宽大于等于该值的加速器。
	MinBandwidthGBps *float32 `json:"minBandwidthGBps,omitempty"`

	// MinMemoryGB 优先选择显存大于等于该值的加速器。
	MinMemoryGB *float32 `json:"minMemoryGB,omitempty"`

	// PrecisionSupport 精度/数据类型支持要求。
	// - required: 必须全部支持的精度类型（硬性要求，缺一不可）。
	// - preferred: 最好支持的精度类型（加分项，支持越多越优先）。
	// 可用于区分模型推理/训练对 INT8、BF16、FP16 等的需求。
	PrecisionSupport *AcceleratorPreferencePrecisionSupport `json:"precisionSupport,omitempty"`

	// PreferHighBandwidth 是否优先高带宽显存。
	PreferHighBandwidth *bool `json:"preferHighBandwidth,omitempty"`

	// PreferredTypes 优先选择的加速器型号列表（如 ["NVIDIA A100", "NVIDIA H100"]），按顺序优先。
	PreferredTypes *[]string `json:"preferredTypes,omitempty"`

	// Weight 偏好权重，数值越大优先级越高。
	Weight *int `json:"weight,omitempty"`
}

// AcceleratorPreferencePrecisionSupportPreferred defines model for AcceleratorPreference.PrecisionSupport.Preferred.
type AcceleratorPreferencePrecisionSupportPreferred string

// AcceleratorPreferencePrecisionSupportRequired defines model for AcceleratorPreference.PrecisionSupport.Required.
type AcceleratorPreferencePrecisionSupportRequired string

type AcceleratorPreferenceAdvanced struct {
	// PcieGen PCIe代数需求（如Gen4、Gen5）
	PcieGen *string `json:"pcieGen,omitempty"`

	// PcieLanes PCIe通道数需求
	PcieLanes *int `json:"pcieLanes,omitempty"`

	// VendorSpecificFeatures 厂商特有高级参数
	VendorSpecificFeatures *map[string]interface{} `json:"vendorSpecificFeatures,omitempty"`
}

type AcceleratorPreferencePrecisionSupport struct {
	// Preferred 最好支持的精度类型（如 FP16，支持越多越优先）。
	Preferred *[]AcceleratorPreferencePrecisionSupportPreferred `json:"preferred,omitempty"`

	// Required 必须全部支持的精度类型（如 INT8、BF16）。
	Required *[]AcceleratorPreferencePrecisionSupportRequired `json:"required,omitempty"`
}

// AffinityPolicy 亲和性策略对象，支持有序降级和补充说明。
// - policies: 有序亲和性策略列表，调度器依次尝试，前者优先，未列出者一律不接受。
// - description: 可选，便于提交时补充说明。
// 典型示例：
//
//	policies: [singleHost, sameSwitchS0]
//	description: "优先同机，不满足则同S0，不接受S1/S2。"
//
// policies 可为空或仅写最后一级（如 sameSwitchS2）代表“无更高亲和性要求”。
type AffinityPolicy struct {
	// Description 可选，补充说明信息。
	Description *string `json:"description,omitempty"`

	// Policies 有序亲和性策略列表，**必须按亲和性强到弱排序，否则无效**。
	// - sameNumaNode: 同 NUMA 节点（最强）
	// - singleHost: 所有资源同一物理机
	// - sameSwitchS0: 同一 S0（ToR/机架）交换机
	// - sameSwitchS1: 同一 S1（minipod）交换机
	// - sameSwitchS2: 同一 S2（bigpod）交换机
	// 调度器依次尝试，前者优先，未列出者一律不接受。
	//
	// **排序要求：必须从强到弱。例如：**
	// 正确：
	//   policies: [sameNumaNode, singleHost, sameSwitchS0]
	// 错误：
	//   policies: [sameSwitchS0, singleHost]  # 错误，强亲和性在后
	//
	// 若顺序错误，API 校验将拒绝。
	//
	// Python 校验片段示例：
	// ```python
	// AFFINITY_ORDER = {
	//     "sameNumaNode": 1,
	//     "singleHost": 2,
	//     "sameSwitchS0": 3,
	//     "sameSwitchS1": 4,
	//     "sameSwitchS2": 5
	// }
	//
	// def validate_affinity_policies(policies):
	//     last = -1
	//     for p in policies:
	//         current = AFFINITY_ORDER.get(p)
	//         if current is None:
	//             return False, f"Unknown affinity: {p}"
	//         if current < last:
	//             return False, f"Affinity policies order error: {policies}"
	//         last = current
	//     return True, ""
	// ```
	//
	// 如发现顺序错误，将返回详细错误信息。
	Policies []AffinityPolicyPolicies `json:"policies"`
}

// AffinityPolicyPolicies defines model for AffinityPolicy.Policies.
type AffinityPolicyPolicies string

// AllocationSegment 分段分配明细结构。表达一次连续分配的资源窗口、集群、节点、卡型、数量、亲和性、NUMA 等实际分配情况。
// 支持弹性调度、断点续训、异构资源等多场景。
type AllocationSegment struct {
	// AcceleratorCategory 卡的类别:gpu/xpu/npu
	AcceleratorCategory string `json:"acceleratorCategory"`

	// AcceleratorIds 实际分配到的加速器ID列表
	AcceleratorIds []string `json:"acceleratorIds"`

	// AcceleratorType 分配到的GPU/加速器卡型（如 "A100", "H100", "V100", "Ascend910" 等）。如为异构分配，可用逗号分隔或数组表达。
	AcceleratorType string `json:"acceleratorType"`

	// AffinityExplanation 亲和性约束满足/降级的详细说明。
	AffinityExplanation *string `json:"affinityExplanation,omitempty"`

	// Allocated 本段是否成功分配
	Allocated bool `json:"allocated"`

	// Cluster 资源实际出让方的集群信息（Cluster，也就是NodeLevel），用于后续归属和资源转移。
	// 包含 zone、dc、physicalCluster、logicalCluster 等。
	Cluster    Cluster     `json:"cluster"`
	CommitInfo *CommitInfo `json:"commitInfo,omitempty"`

	// Count 实际分配的加速器数量（本段总数），可与请求 count 对比。
	Count *int `json:"count,omitempty"`

	// Degraded 是否发生降级（如未满足 preferred/部分软约束）
	Degraded *bool `json:"degraded,omitempty"`

	// Explanation 本段分配的约束满足情况、降级原因等
	Explanation *string `json:"explanation,omitempty"`

	// GroupAffinitySatisfied 是否满足组间亲和性（如多组同机/同集群等）。
	GroupAffinitySatisfied *bool `json:"groupAffinitySatisfied,omitempty"`

	// Id allocationSegment ID
	Id string `json:"id"`

	// NodeIds 实际分配到的节点ID列表
	NodeIds []string `json:"nodeIds"`

	// NumaConfigASatisfied 实际分配的 NUMA 配置是否满足。
	NumaConfigASatisfied *bool `json:"numaConfigASatisfied,omitempty"`

	// NumaConfigExplanation NUMA 配置满足/降级的详细说明
	NumaConfigExplanation *string `json:"numaConfigExplanation,omitempty"`

	// Preemptible 是否可被抢占
	Preemptible *bool `json:"preemptible,omitempty"`

	// ReplicaAffinitySatisfied 是否满足副本间亲和性（如 singleHost、sameNumaNode 等）。
	ReplicaAffinitySatisfied *bool `json:"replicaAffinitySatisfied,omitempty"`

	// Replicas 实际分配的副本数（本段），可与请求 replicas 对比。
	Replicas *int `json:"replicas,omitempty"`

	// SatisfactionGuarantees 本段分配的置信度下可保证的资源满足率（satisfactionGuarantees）。描述在不同置信度（confidence ∈ [0,1]）下，至少可获得的资源比例（rate ∈ [0,1]）。
	// 其中 rate 所对应的资源基数为该 allocationSegment 实际分配所对应的请求资源总量（如 count * replicas），如有多种资源需求，则以主资源（如 GPU/加速器）为准。
	// 例如：{confidence: 0.99, rate: 0.2} 表示有 99% 概率获得至少 20% 请求资源。
	SatisfactionGuarantees *[]AllocationSegmentSatisfactionGuarantee `json:"satisfactionGuarantees,omitempty"`

	// SatisfactionProbabilities 本段分配的资源满足概率表达。描述在不同资源满足率下（rate ∈ [0,1]），获得该满足率的概率（probability ∈ [0,1]）。
	// 其中 rate 所对应的资源基数为该 allocationSegment 实际分配所对应的请求资源总量（如 count * replicas），如有多种资源需求，则以主资源（如 GPU/加速器）为准。
	// 例如：{rate: 0.8, probability: 0.75} 表示有 75% 概率获得至少 80% 的请求资源。
	SatisfactionProbabilities *[]AllocationSegmentSatisfactionProbability `json:"satisfactionProbabilities,omitempty"`

	// TimeWindow 任务调度时间窗口与周期性规则。
	// - 支持一次性窗口（startTime/endTime）
	// - 支持长期服务（endTime 可省略）
	// - 支持周期性窗口（recurrence/rrule + windowDuration）
	// - 所有时间均以 workload.timezone 解释，未指定时区则为 UTC
	// 典型用法：
	// ```yaml
	//   # 一次性训练任务
	//   timeWindow:
	//     startTime: "2025-04-23T02:00:00Z"
	//     endTime:   "2025-04-23T04:00:00Z"
	//   # 长期在线服务
	//   timeWindow:
	//     startTime: "2025-04-23T00:00:00Z"
	//   # 每天凌晨2:00-4:00周期性批量任务
	//   timeWindow:
	//     startTime: "2025-04-23T02:00:00Z"
	//     recurrence: "0 2 * * *"
	//     windowDuration: 7200
	// ```
	TimeWindow TimeWindow `json:"timeWindow"`
}

type AllocationSegmentSatisfactionGuarantee struct {
	// Confidence 置信度（概率，0~1）。
	Confidence *float32 `json:"confidence,omitempty"`

	// Rate 可保证的资源满足率（0~1）。请求总量指本 allocationSegment 对应的加速器总数（count * replicas），如有多种资源需求，则以主资源为准。
	Rate *float32 `json:"rate,omitempty"`
}

type AllocationSegmentSatisfactionProbability struct {
	// Probability 获得该满足率的概率（0~1）。
	Probability *float32 `json:"probability,omitempty"`

	// Rate 资源满足率（占请求总量的比例，0~1）。请求总量指本 allocationSegment 对应的加速器总数（count * replicas），如有多种资源需求，则以主资源为准。
	Rate *float32 `json:"rate,omitempty"`
}

// Cluster 资源实际出让方的集群信息（Cluster，也就是NodeLevel），用于后续归属和资源转移。
// 包含 zone、dc、physicalCluster、logicalCluster 等。
type Cluster struct {
	// Dc 数据中心，如 LF、HL
	Dc *string `json:"dc,omitempty"`

	// LogicalCluster 逻辑集群
	LogicalCluster *string `json:"logicalCluster,omitempty"`

	// Partition partition，如 micro, gpu, socket, yodel
	Partition *string `json:"partition,omitempty"`

	// PhysicalCluster 物理集群
	PhysicalCluster *string `json:"physicalCluster,omitempty"`

	// Zone 区域（如 CN, US 等）
	Zone *string `json:"zone,omitempty"`
}

// CommitInfo defines model for CommitInfo.
type CommitInfo struct {
	// CommitInfoUnits commit info 单元
	CommitInfoUnits *[]CommitInfoUnit `json:"commitInfoUnits,omitempty"`

	// ReservationName 创建的 reservationName
	ReservationName *string `json:"reservationName,omitempty"`

	// ResourcePoolName 在 MatchingIntent 借用方的 resourceGroupId 下 resource pool name，撮合系统 commit 的 Sched-Quota 会通过改 resource pool 出借给借用方。对应 Quota 内的资源池概念。例子: compute-31-hj-bernard.prod-default-default-guarantee
	ResourcePoolName *string `json:"resourcePoolName,omitempty"`

	// ToleranceName 在随着 reservation 创建，调度到具体 node 时，在对应的 node 上写入 taint，调度 pod 时手动指定 tolerance name，可使用对应的 reservation
	ToleranceName *string `json:"toleranceName,omitempty"`

	// TopologyInfo 实际分配到的节点的拓扑信息
	TopologyInfo *map[string]string `json:"topologyInfo,omitempty"`
}

// CommitInfoUnit defines model for CommitInfoUnit.
type CommitInfoUnit struct {
	// Az ToB AZ信息
	Az *string `json:"az,omitempty"`

	// Bigpod ToB bigpod信息
	Bigpod *string `json:"bigpod,omitempty"`

	// FedmemberCluster ToB fedmemberCluster信息
	FedmemberCluster *string `json:"fedmemberCluster,omitempty"`

	// IaasCluster ToB iaasCluster信息
	IaasCluster *string `json:"iaasCluster,omitempty"`

	// MemberNodeLevel ToB 子集群nodeLevel信息
	MemberNodeLevel *string `json:"memberNodeLevel,omitempty"`

	// NodeLevel ToB nodeLevel信息
	NodeLevel *string `json:"nodeLevel,omitempty"`

	// OrderID ToB 订单ID
	OrderID *string `json:"orderID,omitempty"`

	// Quota 对于 ToB 业务，在Fed集群上需要拆分多次qrr创建
	Quota *int `json:"quota,omitempty"`

	// Region ToB region信息
	Region *string `json:"region,omitempty"`

	// ReservationName 创建的 reservationName
	ReservationName *string `json:"reservationName,omitempty"`

	// Specification ToB 规格信息
	Specification *string `json:"specification,omitempty"`

	// ToleranceName 在随着 reservation 创建，调度到具体 node 时，在对应的 node 上写入 taint，调度 pod 时手动指定 tolerance name，可使用对应的 reservation
	ToleranceName *string `json:"toleranceName,omitempty"`

	// TopologyInfo 实际分配到的节点的拓扑信息
	TopologyInfo *map[string]string `json:"topologyInfo,omitempty"`

	// Vrdma ToB vrdma信息
	Vrdma *string `json:"vrdma,omitempty"`
}

// ErrorResponse 通用错误响应结构
type ErrorResponse struct {
	// Code 错误代码
	Code *string `json:"code,omitempty"`

	// Message 错误描述
	Message *string `json:"message,omitempty"`
}

// FlexibleAllocation 灵活分配配置，表示在 startTime 和 endTime 之间的时间窗口内，
// 只要能分配任意连续时间的配额即可，不需要覆盖整个时间窗口。
// 适用于批处理任务，可以在指定的较大时间范围内灵活安排执行时间。
type FlexibleAllocation struct {
	// MaxDuration 最大连续时间长度（小时），表示最多只需要这么长的连续时间段。
	// 用于限制资源申请的最大时长，避免占用过多资源。
	// 如果不指定，则不设置上限，可能会分配整个时间窗口。
	MaxDuration *int `json:"maxDuration,omitempty"`

	// MinDuration 最小连续时间长度（小时），表示至少需要这么长的连续时间段。
	// 如果不指定，默认为整个时间窗口长度。
	MinDuration *int `json:"minDuration,omitempty"`

	// PreferredStartTime （可选）首选开始时间，调度器会尽量安排在这个时间开始，但不保证。
	// 必须在 startTime 和 endTime 之间。
	PreferredStartTime *time.Time `json:"preferredStartTime,omitempty"`

	// Priority 分配优先级策略：
	// - earliest: 优先分配最早可用的时间段
	// - latest: 优先分配最晚可用的时间段
	// - any: 任意可用时间段均可
	Priority *FlexibleAllocationPriority `json:"priority,omitempty"`
}

// FlexibleAllocationPriority 分配优先级策略：
// - earliest: 优先分配最早可用的时间段
// - latest: 优先分配最晚可用的时间段
// - any: 任意可用时间段均可
type FlexibleAllocationPriority string

// GroupResult defines model for GroupResult.
type GroupResult struct {
	// AllocationSegments 推荐/默认分配方案的分段明细。每个 allocationSegment 表达一次连续分配（如断点续训、弹性调度等），支持多段时间窗口。
	// 典型场景：任务被拆分为多个 timeWindow 分配到不同 cluster 或节点。
	AllocationSegments []AllocationSegment `json:"allocationSegments"`

	// Candidates 备选分配方案列表。每个元素为一种可选的资源分配方案（如“100张H100”或“300张H20”），由上游业务/用户自主选择采用哪一种。
	// 每个候选方案结构与主 allocationSegments 完全一致，均为 allocationSegments 数组，复用 AllocationSegment 组件。
	// - 若只返回一个方案，可与 allocationSegments 兼容。
	// - 若有多个方案，allocationSegments 可视为推荐/默认方案，candidates 为全部可选方案。
	Candidates *[]GroupResultCandidate `json:"candidates,omitempty"`

	// GroupIndex 对应请求 groups 的下标
	GroupIndex int `json:"groupIndex"`

	// SatisfactionGuarantees 置信度下可保证的资源满足率（satisfactionGuarantees）。描述在不同置信度（confidence ∈ [0,1]）下，至少可获得的资源比例（rate ∈ [0,1]）。
	// 其中 rate 所对应的资源基数为该 group 在 Intent 中声明的加速器总数（count * replicas），如有多种资源需求，则以主资源（如 GPU/加速器）为准。
	// 例如：{confidence: 0.99, rate: 0.2} 表示有 99% 概率获得至少 20% 请求资源。
	SatisfactionGuarantees *[]GroupResultSatisfactionGuarantee `json:"satisfactionGuarantees,omitempty"`

	// SatisfactionProbabilities 资源满足概率表达。描述在不同资源满足率下（rate ∈ [0,1]），获得该满足率的概率（probability ∈ [0,1]）。
	// 其中 rate 所对应的资源基数为该 group 在 Intent 中声明的加速器总数（count * replicas），如有多种资源需求，则以主资源（如 GPU/加速器）为准。
	// 例如：{rate: 0.8, probability: 0.75} 表示有 75% 概率获得至少 80% 的请求资源。
	SatisfactionProbabilities *[]GroupResultSatisfactionProbability `json:"satisfactionProbabilities,omitempty"`
}

type GroupResultCandidate struct {
	// AllocationSegments 该候选方案的分段分配明细，结构与 groupResults.allocationSegments 相同。
	AllocationSegments []AllocationSegment `json:"allocationSegments"`
}

type GroupResultSatisfactionGuarantee struct {
	// Confidence 置信度（概率，0~1）。
	Confidence *float32 `json:"confidence,omitempty"`

	// Rate 可保证的资源满足率（0~1）。请求总量指该 group 在 Intent 中声明的加速器总数（count * replicas），如有多种资源需求，则以主资源为准。
	Rate *float32 `json:"rate,omitempty"`
}

type GroupResultSatisfactionProbability struct {
	// Probability 获得该满足率的概率（0~1）。
	Probability *float32 `json:"probability,omitempty"`

	// Rate 资源满足率（占请求总量的比例，0~1）。请求总量指该 group 在 Intent 中声明的加速器总数（count * replicas），如有多种资源需求，则以主资源为准。
	Rate *float32 `json:"rate,omitempty"`
}

// GroupResults 多组资源分配明细，对应请求 groups。每组可包含多段分配（如断点续训/弹性调度），每段独立标注 timeWindow、cluster 等。
type GroupResults = []GroupResult

// GroupSpec 单个资源分组的详细需求描述。
// - 用于表达每组独立的副本数、加速器、CPU、NUMA、亲和性、网络等资源约束。
// - replicaAffinity：主分布策略，约束本组内所有副本的资源亲和性（如同机、同 NUMA、同交换机等）。
// - groupAffinity：协同策略，约束本组与其他 group 的资源亲和性（如多组在同一物理机/机架等），仅多组协同场景需要。
// - acceleratorRequirements、numaConfig、network 等用于细化资源和拓扑需求。
// - extraFields：调用方可以传入的其他字段，调度器会原封不动返回；当然，后续的算法可能会使用里面的字段。
// 典型 YAML 示例：
// ```yaml
//
//	groups:
//	  - replicas: 4
//	    gpusPerReplica: 8
//	    replicaAffinity:
//	      policies: [singleHost]
//	    acceleratorPreference:
//	      type: auto
//	      minMemoryGB: 80
//	  - gpusPerReplica: 2
//	    replicaAffinity:
//	      policies: [sameNumaNode]
//	    groupAffinity:
//	      policies: [singleHost]
//	    acceleratorPreference:
//	      type: auto
//	      minMemoryGB: 40
//	    numaConfig:
//	      numaRequired: true
//	      numaNodeCount: 2
//	    network:
//	      minBandwidthGbps: 100
//	      maxHops: 2
//
// ```
// 建议优先通过 replicaAffinity 控制主分布，groupAffinity 仅在多组强协同需求时使用。
type GroupSpec struct {
	// AcceleratorPreference 加速器偏好（软约束），用于表达调度优先选择的加速器型号、特性、能力或权重。
	// - 非硬性要求，调度器会尽量满足偏好，无法满足时可降级。
	// - 适合表达优先型号、带宽、特性、排序权重等。
	// - 仅描述单卡能力/特性，不涉及数量和分布。
	//
	// Accelerator preference (soft constraint), describes preferred accelerator types, features, capabilities, or weights for scheduling. Scheduler will try to satisfy preferences but may downgrade if not possible. Only describes per-card features, not group-level count/distribution.
	//
	// 典型字段示例：
	//   preferredTypes: ["NVIDIA A100", "NVIDIA H100"]
	//   preferHighBandwidth: true
	//   minMemoryGB: 40
	//   weight: 10
	AcceleratorPreference AcceleratorPreference `json:"acceleratorPreference"`

	// Comment 备注信息
	Comment *string `json:"comment,omitempty"`

	// CommitExtraFields 提交时需要传递的额外字段，用于扩展业务逻辑。
	// 这些字段会在撮合过程中保留，并在结果中返回，但不会影响撮合决策
	CommitExtraFields *GroupSpecCommitExtraFields `json:"commitExtraFields,omitempty"`

	// CpuCores 每组/每副本所需的CPU核数，推荐与 acceleratorPreference、numaConfig 等并列表达。
	// 典型表达：
	//   groups:
	//     - gpusPerReplica: 4
	//       cpuCores: 32
	//       acceleratorPreference: ...
	//     - gpusPerReplica: 2
	//       cpuCores: 64
	//       acceleratorPreference: ...
	CpuCores *int `json:"cpuCores,omitempty"`

	// Elasticity 分组级别的弹性伸缩配置。
	Elasticity *GroupSpecElasticity `json:"elasticity,omitempty"`

	// EstimatedDuration 可选。使用此资源配置的预计执行时间（小时）。
	// 具有相近估计时间的不同角色组会被调度器优先组合在一起。
	EstimatedDuration *int `json:"estimatedDuration,omitempty"`

	// GpusPerReplica 每个副本需要的加速器数量
	GpusPerReplica int `json:"gpusPerReplica"`

	// GroupAffinity 亲和性策略对象，支持有序降级和补充说明。
	// - policies: 有序亲和性策略列表，调度器依次尝试，前者优先，未列出者一律不接受。
	// - description: 可选，便于提交时补充说明。
	// 典型示例：
	//   policies: [singleHost, sameSwitchS0]
	//   description: "优先同机，不满足则同S0，不接受S1/S2。"
	// policies 可为空或仅写最后一级（如 sameSwitchS2）代表“无更高亲和性要求”。
	GroupAffinity *AffinityPolicy `json:"groupAffinity,omitempty"`

	// GroupRole 可选。指明此组在整个工作负载中的逻辑角色（如 "trainer", "rollout" 等）。
	// 具有相同 groupRole 的多个组视为同一角色的不同资源配置选项。
	// 调度器将从每个 groupRole 中最多选择一个组进行分配。
	GroupRole *string `json:"groupRole,omitempty"`

	// LocationConstraint 资源位置约束与偏好。支持 zone/dc/cluster 三个维度，均复用 LocationAffinity 组件。
	// - required：必须调度到的区域/集群（硬约束，缺一不可，否则拒绝分配）。
	// - preferred：优先调度到的区域/集群（软约束，按数组顺序表达优先级，无法满足时可降级）。
	// - forbidden：禁止调度到的区域/集群（黑名单，调度器绝不分配）。
	// - cluster 字段统一表达物理/逻辑集群，由业务方约定命名。
	LocationConstraint *LocationConstraint `json:"locationConstraint,omitempty"`

	// Name 分组名称（可选，便于识别和管理），比如对应推理场景，可以是 Service Name
	Name *string `json:"name,omitempty"`

	// Network 网络带宽与跳数约束，仅作用于本组资源。
	// - minBandwidthGbps: 本组内任意两点之间的最小带宽要求。
	// - maxHops: 本组内任意两点之间的最大网络跳数（推荐与 replicaAffinity 联合使用）。
	// - rdma: RDMA 网络需求类型（none/any/infiniband/roce/iwarp，默认 none）。
	// 建议只在确实需要网络带宽/距离/高性能网络时填写，物理/拓扑分布请用 replicaAffinity 表达。
	// 典型 YAML 示例：
	//   network:
	//     minBandwidthGbps: 200
	//     maxHops: 2
	//     rdma: infiniband
	Network *GroupSpecNetwork `json:"network,omitempty"`

	// NumaConfig NUMA 相关配置，仅作用于本组资源。用于细粒度约束如节点数、本地内存、CPU 绑定等。
	NumaConfig *NUMAConfig `json:"numaConfig,omitempty"`

	// OptionId 可选。当组被视为某个角色的资源选项时，此字段提供选项的唯一标识符。
	// 通常与 groupRole 一起使用。
	OptionId *string `json:"optionId,omitempty"`

	// OptionRank 可选。此选项在同角色组中的排序等级，数值越小越优先考虑。
	OptionRank *int `json:"optionRank,omitempty"`

	// ReplicaAffinity 亲和性策略对象，支持有序降级和补充说明。
	// - policies: 有序亲和性策略列表，调度器依次尝试，前者优先，未列出者一律不接受。
	// - description: 可选，便于提交时补充说明。
	// 典型示例：
	//   policies: [singleHost, sameSwitchS0]
	//   description: "优先同机，不满足则同S0，不接受S1/S2。"
	// policies 可为空或仅写最后一级（如 sameSwitchS2）代表“无更高亲和性要求”。
	ReplicaAffinity AffinityPolicy `json:"replicaAffinity"`

	// Replicas 该资源分组的副本数量。
	// 例如，对于分布式训练，可以定义一个 trainers 组（replicas=8）和一个 ps 组（replicas=1）。
	Replicas *int `json:"replicas,omitempty"`

	// TobConfig 针对 ToB 业务的特殊配置字段，如vendor, region, az等。
	// 建议仅在 ToB 业务中使用。
	TobConfig *VolcConfig `json:"tobConfig,omitempty"`

	// TobReplicas tob侧传入的原始副本数。
	TobReplicas *int `json:"tobReplicas,omitempty"`

	// TopologyConstraint TopologyConstraint 可选。指明具体的拓扑值，例如cluster=xx, nodeLevel=yy,
	// tce.kubernetes.io/rdmaminipod=zz，拓扑撮合时便会去预留满足这些拓扑条件的节点。
	TopologyConstraint *map[string]string `json:"topologyConstraint,omitempty"`

	// VolcConfig 针对 ToB 业务的特殊配置字段，如vendor, region, az等。
	// 建议仅在 ToB 业务中使用。
	VolcConfig *VolcConfig `json:"volcConfig,omitempty"`

	// VolcReplicas tob侧传入的原始副本数。
	VolcReplicas *int `json:"volcReplicas,omitempty"`
}

func (groupSpec *GroupSpec) IsSameNodeLevel() bool {
	clusterMaxLocations := groupSpec.LocationConstraint.Cluster.MaxLocations
	zoneMaxLocations := groupSpec.LocationConstraint.Zone.MaxLocations
	dcMaxLocations := groupSpec.LocationConstraint.Dc.MaxLocations

	// 默认行为为无需 SameNodeLevel
	if clusterMaxLocations == nil || zoneMaxLocations == nil || dcMaxLocations == nil {
		return false
	}

	return *clusterMaxLocations == 1 && *zoneMaxLocations == 1 && *dcMaxLocations == 1
}

type GroupSpecCommitExtraFields struct {
	// Namespace 在创建资源池时需要创建 k8s queue 对象，namespace 和底层 k8s namespace（集群维度） 一致。
	// 如果传了底下 k8s namespace 不存在的 namespace 就会报错。
	// Bernard：default
	// Arnold：arnold
	Namespace *string `json:"namespace,omitempty"`

	// Qos shared or dedicated
	Qos *string `json:"qos,omitempty"`

	// QueueName 创建资源池的需要，默认是 default。
	QueueName *string `json:"queueName,omitempty"`

	// ReserveTo 创建 Reservation需要
	// 用于 workload 部署对应到具体的 reserveTo，才能使用起来
	// bernard: bernard_to_uce
	// arnold: arnold_to_uce
	ReserveTo *string `json:"reserveTo,omitempty"`
}

type GroupSpecElasticity struct {
	// MaxReplicas 该分组最多可以扩展到的副本数。用于机会性地使用额外资源。
	// 如果未指定，默认为 `replicas` 字段的值，表示不允许扩容。
	// 必须大于或等于 `replicas`。
	MaxReplicas *int `json:"maxReplicas,omitempty"`

	// MinReplicas 该分组最少需要满足的副本数。调度器将保证至少分配 `minReplicas` 个副本。
	// 如果未指定，默认为 `replicas` 字段的值，表示不允许缩容。
	// 必须小于或等于 `replicas`。
	MinReplicas *int `json:"minReplicas,omitempty"`
}

type GroupSpecNetwork struct {
	// MaxHops 最大网络跳数。通常推荐通过 replicaAffinity 控制物理分布，maxHops 仅用于补充软约束。
	MaxHops *int `json:"maxHops,omitempty"`

	// MinBandwidthGbps 最小网络带宽需求（Gbps）。如无特殊需求可不填。
	MinBandwidthGbps *float32 `json:"minBandwidthGbps,omitempty"`

	// Rdma RDMA 网络需求类型。
	// - none: 不需要 RDMA
	// - any: 需要 RDMA，但不限底层实现
	// - infiniband: 必须是 InfiniBand 网络
	// - roce: 必须是 RoCE 网络
	// - iwarp: 必须是 iWARP 网络
	Rdma *GroupSpecNetworkRdma `json:"rdma,omitempty"`

	// StorageConnectivity 存储系统可达性需求，描述是否需要访问特定分布式存储（如 ByteNaS、HDFS）。
	// - byteNaS: 是否要求能访问 ByteNaS 存储
	// - hdfs: 是否要求能访问 HDFS 存储
	// - other: 其他需要可达的存储系统（如 Ceph、OSS 等）
	// 示例：
	//   storageConnectivity:
	//     byteNaS: true
	//     hdfs: false
	//     other: ["Ceph", "OSS"]
	StorageConnectivity *GroupSpecNetworkStorageConnectivity `json:"storageConnectivity,omitempty"`
}

type GroupSpecNetworkStorageConnectivity struct {
	// ByteNaS 是否要求能访问 ByteNaS 存储
	ByteNaS *bool `json:"byteNaS,omitempty"`

	// Hdfs 是否要求能访问 HDFS 存储
	Hdfs *bool `json:"hdfs,omitempty"`

	// Other 其他需要可达的存储系统名称列表
	Other *[]string `json:"other,omitempty"`
}

// GroupSpecNetworkRdma RDMA 网络需求类型。
// - none: 不需要 RDMA
// - any: 需要 RDMA，但不限底层实现
// - infiniband: 必须是 InfiniBand 网络
// - roce: 必须是 RoCE 网络
// - iwarp: 必须是 iWARP 网络
type GroupSpecNetworkRdma string

// LocationAffinity 区域/集群等资源的约束集合，三类数组表达 required/preferred/forbidden。
// - required：必须调度到的资源（硬约束，缺一不可）
// - preferred：优先调度到的资源（软约束，顺序表达优先级）
// - forbidden：禁止调度到的资源（黑名单）
// 可用于 zone、dc、cluster 等多种资源类型的约束表达。
type LocationAffinity struct {
	// Forbidden 禁止调度到的资源（黑名单）
	Forbidden *[]string `json:"forbidden,omitempty"`

	// MaxLocations 最多允许从多少个 location 分配资源。缺省/不填表示不做限制。暂时只支持 1
	MaxLocations *int `json:"maxLocations"`

	// Preferred 优先调度到的资源（软约束，顺序表达优先级）
	Preferred *[]string `json:"preferred,omitempty"`

	// Required 必须调度到的资源（硬约束，缺一不可）
	Required *[]string `json:"required,omitempty"`
}

// LocationConstraint 资源位置约束与偏好。支持 zone/dc/cluster 三个维度，均复用 LocationAffinity 组件。
// - required：必须调度到的区域/集群（硬约束，缺一不可，否则拒绝分配）。
// - preferred：优先调度到的区域/集群（软约束，按数组顺序表达优先级，无法满足时可降级）。
// - forbidden：禁止调度到的区域/集群（黑名单，调度器绝不分配）。
// - cluster 字段统一表达物理/逻辑集群，由业务方约定命名。
type LocationConstraint struct {
	// Cluster 区域/集群等资源的约束集合，三类数组表达 required/preferred/forbidden。
	// - required：必须调度到的资源（硬约束，缺一不可）
	// - preferred：优先调度到的资源（软约束，顺序表达优先级）
	// - forbidden：禁止调度到的资源（黑名单）
	// 可用于 zone、dc、cluster 等多种资源类型的约束表达。
	Cluster *LocationAffinity `json:"cluster,omitempty"`

	// Dc 区域/集群等资源的约束集合，三类数组表达 required/preferred/forbidden。
	// - required：必须调度到的资源（硬约束，缺一不可）
	// - preferred：优先调度到的资源（软约束，顺序表达优先级）
	// - forbidden：禁止调度到的资源（黑名单）
	// 可用于 zone、dc、cluster 等多种资源类型的约束表达。
	Dc *LocationAffinity `json:"dc,omitempty"`

	// SupplyDomain 区域/集群等资源的约束集合，三类数组表达 required/preferred/forbidden。
	// - required：必须调度到的资源（硬约束，缺一不可）
	// - preferred：优先调度到的资源（软约束，顺序表达优先级）
	// - forbidden：禁止调度到的资源（黑名单）
	// 可用于 zone、dc、cluster 等多种资源类型的约束表达。
	SupplyDomain *LocationAffinity `json:"supplyDomain,omitempty"`

	// Zone 区域/集群等资源的约束集合，三类数组表达 required/preferred/forbidden。
	// - required：必须调度到的资源（硬约束，缺一不可）
	// - preferred：优先调度到的资源（软约束，顺序表达优先级）
	// - forbidden：禁止调度到的资源（黑名单）
	// 可用于 zone、dc、cluster 等多种资源类型的约束表达。
	Zone *LocationAffinity `json:"zone,omitempty"`
}

// MatchingIntent 资源撮合意向单主结构，支持多组资源、全局调度参数等。
// - timeWindow、elasticityOptions 等字段为全局调度/弹性参数，作用于整个意向单（所有 group）。
// - serviceId、transitionMinutes 等字段用于支持连续服务的资源过渡。
type MatchingIntent struct {
	// BookWithReservation 对一些少数特殊的需求，需要在 Book的时候就要完成 Reservation
	BookWithReservation *bool `json:"bookWithReservation,omitempty"`

	// CommitDeadline 最晚进行 Quota/资源交付的时间点（Unix 时间戳，单位秒，支持毫秒自动识别）。
	// - 支持传入秒（如 1745630400）或毫秒（如 1745630400000），服务端会自动识别并兼容。
	// - 推荐使用秒级时间戳。
	// 规则：
	//   - 如果为空，  - 如果为空，需要通过 api/match/:id/commit来主动 commit；或者保留旧版的自己来调用 ByteQuota 的 increase 接口自己来申请；commitDeadline为空，且超过一定时间没有 commit，就会进入 cancelled
	//   - 如果commitDealine==0，会立即从 Booked转移到 Commiting状态。
	//   - 如果是其他非0的值，且commitDeadline < decisionDeadline，视为错误，无法通过 validate。
	CommitDeadline *int64 `json:"commitDeadline,omitempty"`

	// DecisionDeadline 最晚决策时间点（Unix 时间戳，单位秒，支持毫秒自动识别）。用于异步模式下指定调度超时时间，超时后系统会返回当前最优结果。
	// - 支持传入秒（如 1745630400）或毫秒（如 1745630400000），服务端会自动识别并兼容。
	// - 推荐使用秒级时间戳。
	// - decisionDealine如果不设置的话，是commitDeadline - 5分钟（如果commitDeadline也没有设置，会取commitDeadline默认值）
	DecisionDeadline *int64 `json:"decisionDeadline,omitempty"`

	// Description 撮合单描述
	Description *string `json:"description,omitempty"`

	// ElasticityOptions 全局弹性策略配置。定义整个意向单在资源不足时的整体行为偏好。
	// 更细粒度的弹性需求（如副本数范围）应在每个资源组的 `elasticity` 字段中定义。
	ElasticityOptions *MatchingIntentElasticityOptions `json:"elasticityOptions,omitempty"`
	ExtraFields       *map[string]interface{}          `json:"extraFields,omitempty"`

	// Groups 资源分组需求列表。每个 group 表示一组同构或异构的加速器资源请求：
	// - 可通过 replicas 指定该组副本数（默认为 1），每个副本的资源需求相同。
	// - 其余字段（count, placementPolicy, acceleratorRequirements）描述每个副本的详细需求。
	// - 适用于多实例/多副本训练、批量推理、异构分组等复杂调度场景。
	// 示例：
	//   groups:
	//     # 4 个副本，每个副本需要 8 张卡，单机亲和
	//     - replicas: 4
	//       count: 8
	//       replicaAffinity: [singleHost]
	//       acceleratorPreference:
	//         type: auto
	//         minMemoryGB: 80
	//     # 单组需求（replicas 可省略，默认为 1）
	//     - count: 2
	//       replicaAffinity: [sameNumaNode]
	//       acceleratorPreference:
	//         type: auto
	//         minMemoryGB: 40
	//       numaConfig:
	//         $ref: '#/components/schemas/NUMAConfig'
	Groups *[]GroupSpec `json:"groups,omitempty"`

	// Name 撮合单名称
	Name *string `json:"name,omitempty"`

	// Requester 请求方信息，包含业务线、资源组等标识。
	// - businessLineName ：业务线名称
	// - businessLineId ：业务线ID
	// - babiUnit: babi 业务线名称
	// - subBabiUnit: babi 子业务线名称
	// - resourceGroupId ：资源组ID
	// - platform ：业务平台标识
	Requester *Requester `json:"requester,omitempty"`

	// ServiceId 服务ID，用于标识连续服务。
	// 同一服务的多个意向单共享相同的 serviceId，便于查询服务的资源分配历史。
	ServiceId *string `json:"serviceId,omitempty"`

	// Status MatchingIntent 的状态枚举，详见 docs/matching_intent_state_machine.md
	// - Matching.Pending
	// - Matching.Provisional
	// - Matching.Booking
	// - Matching.Booked
	// - Matching.Committing
	// - Succeed
	// - Finished
	// - Failed
	// - Canceling
	// - Canceled
	Status *MatchingIntentStatus `json:"status,omitempty"`

	// TimeWindow 任务调度时间窗口与周期性规则。
	// - 支持一次性窗口（startTime/endTime）
	// - 支持长期服务（endTime 可省略）
	// - 支持周期性窗口（recurrence/rrule + windowDuration）
	// - 所有时间均以 workload.timezone 解释，未指定时区则为 UTC
	// 典型用法：
	// ```yaml
	//   # 一次性训练任务
	//   timeWindow:
	//     startTime: "2025-04-23T02:00:00Z"
	//     endTime:   "2025-04-23T04:00:00Z"
	//   # 长期在线服务
	//   timeWindow:
	//     startTime: "2025-04-23T00:00:00Z"
	//   # 每天凌晨2:00-4:00周期性批量任务
	//   timeWindow:
	//     startTime: "2025-04-23T02:00:00Z"
	//     recurrence: "0 2 * * *"
	//     windowDuration: 7200
	// ```
	TimeWindow *TimeWindow `json:"timeWindow,omitempty"`

	// TransitionMinutes 资源过渡时间（分钟），用于指定新旧资源的重叠时间。
	// 系统会确保新旧资源在这段时间内同时可用，以便服务平滑迁移。
	// 默认为 60 分钟。
	TransitionMinutes *int `json:"transitionMinutes,omitempty"`

	// Workload 工作负载业务意图描述。表达任务类型、优先级、调度方式、业务标签等元信息，不涉及资源规格。
	Workload *Workload `json:"workload,omitempty"`
}

// MatchingIntentStatus MatchingIntent 的状态枚举，详见 docs/matching_intent_state_machine.md
// - Matching.Pending
// - Matching.Provisional
// - Matching.Booking
// - Matching.Booked
// - Matching.Committing
// - Succeed
// - Finished
// - Failed
// - Canceling
// - Canceled
type MatchingIntentStatus string

type MatchingIntentElasticityOptions struct {
	// AllowAlternativeTypes 是否允许系统在主加速器类型不可用时，自动尝试匹配 `group.acceleratorRequirements.alternativeTypes` 中定义的可替代加速器类型。
	// - `true`: 允许自动匹配替代类型。
	// - `false`: (默认) 严格要求匹配主加速器类型。
	AllowAlternativeTypes *bool `json:"allowAlternativeTypes,omitempty"`

	// DownscalingPriority 定义资源缩容时的分组优先级。数组中的分组名称（group.name）按顺序排列，排在前面的分组将优先被缩容。
	// 例如：["spot-workers", "regular-workers"]，表示在需要缩容时，系统会先尝试缩减 "spot-workers" 组的副本数。
	// 如果为空或未指定，系统将根据默认策略（如成本、资源类型等）进行缩容。
	DownscalingPriority *[]string `json:"downscalingPriority,omitempty"`
}

// MatchingResult 撮合结果结构，包含分配详情、拓扑、原因解释、评分、诊断等。
// status 字段支持 pending（初始）、provisional（中间可用解）、success/failed/partial（最终解）。
// provisional 状态下，groupResults 和 candidates 可返回当前可行解，explanation 字段说明置信度。
// 对于连续服务，一个意向单可能有多个结果，按时间窗口排序，支持资源平滑过渡。
type MatchingResult struct {
	// Diagnostics 撮合诊断信息（如警告、提示等）
	Diagnostics *[]string `json:"diagnostics,omitempty"`

	// Explanation 当前分配状态的说明（如 provisional 时可解释置信度、变更风险等）
	Explanation *string `json:"explanation,omitempty"`

	// GroupResults 多组资源分配明细，对应请求 groups。每组可包含多段分配（如断点续训/弹性调度），每段独立标注 timeWindow、cluster 等。
	GroupResults *GroupResults `json:"groupResults,omitempty"`

	// MatchId 撮合任务ID
	MatchId string `json:"matchId"`

	// MatchingScore 撮合评分（可选）
	MatchingScore *float32 `json:"matchingScore,omitempty"`

	// SelectedOptions 当使用 groupRole 标识组角色时，此字段列出调度器从每个角色中选择的选项ID（optionId）。
	// groupResults 中只会包含这些被选中的组的结果。
	SelectedOptions *[]string `json:"selectedOptions,omitempty"`

	// Status 撮合状态：
	// - pending: 初始，尚无可行解
	// - booking: 预留中
	// - booked: 已预留，待提交，等待 commitDeadline，或者 commit 操作提交流转到 committing
	// - committing: 提交中
	// - success: 最终分配成功（终态）
	// - failed: 最终分配失败（终态）
	// - cancelling: matching单再主动取消中
	// - cancelled: matching单被主动取消（终态）
	// - partial: 部分分配成功（暂未实现）
	// - provisional: 有可行但非最优解，可多次更新（暂未实现）
	Status MatchingResultStatus `json:"status"`

	// TimeWindow 任务调度时间窗口与周期性规则。
	// - 支持一次性窗口（startTime/endTime）
	// - 支持长期服务（endTime 可省略）
	// - 支持周期性窗口（recurrence/rrule + windowDuration）
	// - 所有时间均以 workload.timezone 解释，未指定时区则为 UTC
	// 典型用法：
	// ```yaml
	//   # 一次性训练任务
	//   timeWindow:
	//     startTime: "2025-04-23T02:00:00Z"
	//     endTime:   "2025-04-23T04:00:00Z"
	//   # 长期在线服务
	//   timeWindow:
	//     startTime: "2025-04-23T00:00:00Z"
	//   # 每天凌晨2:00-4:00周期性批量任务
	//   timeWindow:
	//     startTime: "2025-04-23T02:00:00Z"
	//     recurrence: "0 2 * * *"
	//     windowDuration: 7200
	// ```
	TimeWindow *TimeWindow `json:"timeWindow,omitempty"`

	// Timestamp 撮合完成时间
	Timestamp *time.Time `json:"timestamp,omitempty"`
}

// MatchingResultStatus 撮合状态：
// - pending: 初始，尚无可行解
// - booking: 预留中
// - booked: 已预留，待提交，等待 commitDeadline，或者 commit 操作提交流转到 committing
// - committing: 提交中
// - success: 最终分配成功（终态）
// - failed: 最终分配失败（终态）
// - cancelling: matching单再主动取消中
// - cancelled: matching单被主动取消（终态）
// - partial: 部分分配成功（暂未实现）
// - provisional: 有可行但非最优解，可多次更新（暂未实现）
type MatchingResultStatus string

// MatchingStrategy 撮合策略结构（占位，后续完善）
type MatchingStrategy = map[string]interface{}

// NUMAConfig NUMA 相关配置，仅作用于本组资源。用于细粒度约束如节点数、本地内存、CPU 绑定等。
type NUMAConfig struct {
	// CpuPinning 是否要求CPU绑定
	CpuPinning *bool `json:"cpuPinning,omitempty"`

	// NumaAware 是否要求NUMA拓扑感知
	NumaAware *bool `json:"numaAware,omitempty"`

	// NumaLocalMemoryGB 每个NUMA节点本地内存需求（GB）
	NumaLocalMemoryGB *float32 `json:"numaLocalMemoryGB,omitempty"`

	// NumaNodeCount 需要的NUMA节点数
	NumaNodeCount *int `json:"numaNodeCount,omitempty"`

	// NumaOptimizedInterconnect 是否要求NUMA优化互联
	NumaOptimizedInterconnect *bool `json:"numaOptimizedInterconnect,omitempty"`

	// NumaRequired 是否必须支持NUMA架构
	NumaRequired *bool `json:"numaRequired,omitempty"`
}

// Requester 请求方信息，包含业务线、资源组等标识。
// - businessLineName ：业务线名称
// - businessLineId ：业务线ID
// - babiUnit: babi 业务线名称
// - subBabiUnit: babi 子业务线名称
// - resourceGroupId ：资源组ID
// - platform ：业务平台标识
type Requester struct {
	// BabiUnit babi 业务线名称
	// Optional
	BabiUnit *string `json:"babiUnit,omitempty"`

	// BillTree billTree
	// Optional
	BillTree *string `json:"billTree,omitempty"`

	// BusinessLineId 业务线唯一标识
	// Required
	BusinessLineId string `json:"businessLineId"`

	// BusinessLineName 业务线名称
	// Required
	BusinessLineName string `json:"businessLineName"`

	// Platform 业务平台标识。用于区分资源请求来源平台。
	// 枚举值：Bernard、Arnold、Engine-Serving、Ve-MaaS、Ve-IaaS、Ve-MLP、Engine-Training、。
	// Required
	Platform RequesterPlatform `json:"platform"`

	// ResourceGroupId 资源组唯一标识
	// Required
	ResourceGroupId string `json:"resourceGroupId"`

	// ResourceGroupName 资源组名
	// Optional
	ResourceGroupName *string `json:"resourceGroupName,omitempty"`

	// SubBabiUnit babi 子业务线名称
	// Optional
	SubBabiUnit *string `json:"subBabiUnit,omitempty"`
}

// RequesterPlatform 业务平台标识。用于区分资源请求来源平台。
// 枚举值：Bernard、Arnold、Engine-Serving、Ve-MaaS、Ve-IaaS、Ve-MLP、Engine-Training、。
type RequesterPlatform string

// TimeWindow 任务调度时间窗口与周期性规则。
// - 支持一次性窗口（startTime/endTime）
// - 支持长期服务（endTime 可省略）
// - 支持周期性窗口（recurrence/rrule + windowDuration）
// - 所有时间均以 workload.timezone 解释，未指定时区则为 UTC
// 典型用法：
// ```yaml
//
//	# 一次性训练任务
//	timeWindow:
//	  startTime: "2025-04-23T02:00:00Z"
//	  endTime:   "2025-04-23T04:00:00Z"
//	# 长期在线服务
//	timeWindow:
//	  startTime: "2025-04-23T00:00:00Z"
//	# 每天凌晨2:00-4:00周期性批量任务
//	timeWindow:
//	  startTime: "2025-04-23T02:00:00Z"
//	  recurrence: "0 2 * * *"
//	  windowDuration: 7200
//
// ```
type TimeWindow struct {
	// EndTime 任务/服务可调度的结束时间（ISO 8601 格式）。
	// 对于长期服务可省略，表示“直到主动释放”。
	EndTime *time.Time `json:"endTime,omitempty"`

	// FlexibleAllocation 灵活分配配置，表示在 startTime 和 endTime 之间的时间窗口内，
	// 只要能分配任意连续时间的配额即可，不需要覆盖整个时间窗口。
	// 适用于批处理任务，可以在指定的较大时间范围内灵活安排执行时间。
	FlexibleAllocation *FlexibleAllocation `json:"flexibleAllocation,omitempty"`

	// Recurrence （可选）周期性规则，cron 表达式或 RFC 5545 RRULE 格式。
	// 如 "0 2 * * *" 表示每天凌晨2点，"FREQ=WEEKLY;BYDAY=MO" 表示每周一。
	// 仅周期性任务需要填写。
	Recurrence *string `json:"recurrence,omitempty"`

	// StartTime 任务/服务可调度的起始时间（ISO 8601 格式）。
	// 对于周期性任务，表示首次调度的起点。
	StartTime time.Time `json:"startTime"`

	// Timezone （可选）覆盖 workload.timezone，单独指定此 timeWindow 的时区（IANA/Olson 格式）。
	Timezone *string `json:"timezone,omitempty"`

	// WindowDuration （可选）每次周期性任务的持续时长（秒），与 recurrence 配合使用。
	WindowDuration *int `json:"windowDuration,omitempty"`
}

func (tw TimeWindow) Duration() time.Duration {
	return (tw.EndTime).Sub(tw.StartTime)
}

// ValidationResult 校验结果结构
type ValidationResult struct {
	// Errors 校验错误信息列表
	Errors *[]string `json:"errors,omitempty"`

	// IsValid 是否通过校验
	IsValid *bool `json:"isValid,omitempty"`
}

// VolcConfig 针对 ToB 业务的特殊配置字段，如vendor, region, az等。
// 建议仅在 ToB 业务中使用。
type VolcConfig struct {
	// Azs tob可用区名称
	Azs *[]string `json:"azs,omitempty"`

	// Bigpod bigpod名称
	Bigpod *string `json:"bigpod,omitempty"`

	// IaasCluster iaas集群名称
	IaasCluster *string `json:"iaasCluster,omitempty"`

	// Region tob地区名称
	Region *string `json:"region,omitempty"`

	// Specifications 机器规格
	Specifications *[]string `json:"specifications,omitempty"`

	// Vendor vendor 名称
	Vendor *string `json:"vendor,omitempty"`

	// Vrdma 是否开启vrdma
	Vrdma *string `json:"vrdma,omitempty"`
}

// Workload 工作负载业务意图描述。表达任务类型、优先级、调度方式、业务标签等元信息，不涉及资源规格。
type Workload struct {
	// CustomTags 用户自定义标签，支持任意字符串，便于业务扩展/特殊标记。
	CustomTags *[]string `json:"customTags,omitempty"`

	// Description 任务简要说明
	Description *string `json:"description,omitempty"`

	// GpuUtilDailyPast4Weeks 过去 4 周每天的天均 GPU Util（带日期），用于历史利用率分析和智能调度。
	GpuUtilDailyPast4Weeks *[]WorkloadGpuUtilDaily `json:"gpuUtilDailyPast4Weeks,omitempty"`

	// GpuUtilHourlyPast72h 最近 72 小时每小时的 GPU Util（带时间戳），用于短期利用率分析和弹性资源撮合。
	GpuUtilHourlyPast72h *[]WorkloadGpuUtilHourly `json:"gpuUtilHourlyPast72h,omitempty"`

	// JobPolicy 通用作业调度策略（jobPolicy），主要用于 training 场景（如批量训练、弹性调度、生命周期管理），也兼容特殊 serving/data-processing 需求。
	// - elasticityPolicy：声明任务弹性与生命周期能力（如弹性伸缩、挂起/恢复、重启等）。
	// - minRunTime：任务期望的最小连续运行时长（秒）。
	JobPolicy *WorkloadJobPolicy `json:"jobPolicy,omitempty"`

	// Priority 数字型优先级，越大越高。建议范围 1~100。
	Priority int `json:"priority"`

	// PriorityDescription 优先级解释，如 priority 80 ，需要给用户解释清楚为何是80，中文描述
	PriorityDescription *string `json:"priority_description,omitempty"`

	// PriorityDescriptionI18n 优先级解释，如 priority 80 ，需要给用户解释清楚为何是80，英文描述
	PriorityDescriptionI18n *string `json:"priority_description_i18n,omitempty"`

	// Renewal 续期信息，用于表达资源续期和迁移需求。
	// - 支持异构资源替换
	// - 支持迁移提前量和过渡期资源配置
	// - 默认采用滚动迁移策略，确保服务可用性
	Renewal *WorkloadRenewal `json:"renewal,omitempty"`

	// Scene 任务主场景（互斥，必选）：
	//   - training        # 训练（包括全量训练、微调、预训练等）
	//   - serving         # 在线/离线推理服务（模型部署、API服务等）
	//   - data-processing # 数据处理/特征提取/ETL等
	Scene WorkloadScene `json:"scene"`

	// Schedule 可选，支持多日历类型（如工作日、周末、节假日）分别配置弹性调度规则。
	// 每个元素对应一种日历类型的调度策略。
	// training 场景通常无需填写。
	Schedule *[]WorkloadSchedule `json:"schedule"`

	// Tags 官方推荐标签（如 fine-tune、nlp、ai4science、urgent 等），表达业务属性、细分场景。
	Tags *[]string `json:"tags,omitempty"`

	// Timezone 时区，IANA/Olson 格式（如 "Asia/Shanghai"），用于解释调度日历规则和所有时间相关字段。
	// 推荐使用标准时区名，兼容夏令时和国际化业务场景。
	// 默认值："UTC"。如需本地化请显式指定。
	// 典型示例：
	//   timezone: "Asia/Shanghai"  # 北京时间
	//   timezone: "UTC"            # 世界标准时间
	//   timezone: "America/Los_Angeles"  # 美国西部
	Timezone *string `json:"timezone,omitempty"`
}

// WorkloadScene 任务主场景（互斥，必选）：
//   - training        # 训练（包括全量训练、微调、预训练等）
//   - serving         # 在线/离线推理服务（模型部署、API服务等）
//   - data-processing # 数据处理/特征提取/ETL等
type WorkloadScene string

// WorkloadScheduleCalendar 日历规则，指定本条调度规则适用的日期类型：
//   - workday  工作日
//   - weekend  周末
//   - holiday  法定节假日
//   - everyday 每天
type WorkloadScheduleCalendar string

// WorkloadScheduleTimeSegmentsPeakLevel 业务高低峰标识（可选，便于业务解读与监控，也便于调度器按业务高低峰分配资源）。
// - high/medium/low：业务自定义的高、中、低峰时段。
// - zero：该时段无需任何资源，调度器可 scale down 到 0，适用于夜间/节假日无业务需求时自动释放资源。
type WorkloadScheduleTimeSegmentsPeakLevel string

type WorkloadGpuUtilDaily struct {
	// Date 日期（ISO 8601，yyyy-mm-dd）
	Date time.Time `json:"date"`

	// Util GPU 利用率百分比（0~100）
	Util float32 `json:"util"`
}

type WorkloadGpuUtilHourly struct {
	// Datetime 小时起始时间（ISO 8601，yyyy-mm-ddTHH:00:00Z）
	Datetime time.Time `json:"datetime"`

	// Util GPU 利用率百分比（0~100）
	Util float32 `json:"util"`
}

type WorkloadJobPolicy struct {
	// ElasticityPolicy 弹性与生命周期能力声明，调度器据此决定是否可对任务执行伸缩、挂起、恢复、重启等操作。
	// - scaling：是否支持弹性伸缩（scale up/down）。
	// - suspend：是否支持被挂起/恢复（suspend/resume）。
	// - restart：是否支持被重启（restart）。
	// 可根据业务需要扩展更多能力（如 migration、checkpoint 等）。
	ElasticityPolicy *WorkloadElasticityPolicy `json:"elasticityPolicy,omitempty"`

	// MinRunTime 任务期望的最小连续运行时长（秒），调度器会尽量保障任务不中断运行，也可用于表达希望获得最大资源量的最短保障窗口（如“前4小时保障100卡”）。
	MinRunTime *int `json:"minRunTime,omitempty"`
}

type WorkloadElasticityPolicy struct {
	// Restart 是否支持被重启（restart），适用于支持断点续训、幂等等场景。
	Restart *bool `json:"restart,omitempty"`

	// Scaling 是否支持弹性伸缩（如资源缩减/扩容），比如从 100卡收缩到50卡时候 Job 可以自动适应而无需退出。
	// 调度器可在资源紧张时动态调整资源，Job 能自适应，以保证Job的连续运行。
	Scaling *bool `json:"scaling,omitempty"`

	// Suspend 是否支持被挂起和恢复（suspend/resume），调度器可在资源压力大时挂起任务，资源充裕时恢复。
	Suspend *bool `json:"suspend,omitempty"`
}

type WorkloadRenewal struct {
	// MigrationStartOffset 迁移提前量（分钟），表示在正式续期时间点前多久开始迁移过程。
	// 例如：60表示提前1小时开始迁移。
	MigrationStartOffset *int `json:"migrationStartOffset,omitempty"`

	// TransitionResources 迁移过渡期的资源配置，用于描述迁移过程中的额外资源需求。
	TransitionResources *WorkloadTransitionResource `json:"transitionResources,omitempty"`
}

type WorkloadTransitionResource struct {
	// MaxExtraResources 迁移过程中的最大额外资源数量（以首选类型计）。
	// 例如：20表示迁移期间最多额外使用20个资源单位。
	MaxExtraResources *int `json:"maxExtraResources,omitempty"`

	// OverProvisionPercent 迁移过程中的资源超配百分比。
	// 例如：30表示迁移期间需要额外30%的资源。
	OverProvisionPercent *int `json:"overProvisionPercent,omitempty"`
}

type WorkloadSchedule struct {
	// Calendar 日历规则，指定本条调度规则适用的日期类型：
	//   - workday  工作日
	//   - weekend  周末
	//   - holiday  法定节假日
	//   - everyday 每天
	Calendar *WorkloadScheduleCalendar `json:"calendar,omitempty"`

	// TimeSegments 一天内的时段划分及弹性策略。
	// 可按需细分高峰、低峰等时段。
	TimeSegments *[]WorkloadTimeSegment `json:"timeSegments,omitempty"`
}

type WorkloadTimeSegment struct {
	// End 时段结束时间（24小时制，HH:mm）
	End *string `json:"end,omitempty"`

	// PeakLevel 业务高低峰标识（可选，便于业务解读与监控，也便于调度器按业务高低峰分配资源）。
	// - high/medium/low：业务自定义的高、中、低峰时段。
	// - zero：该时段无需任何资源，调度器可 scale down 到 0，适用于夜间/节假日无业务需求时自动释放资源。
	PeakLevel *WorkloadScheduleTimeSegmentsPeakLevel `json:"peakLevel,omitempty"`

	// Scale 资源倍率（可选，默认 1.0）。
	// 表示本时段实际资源量 = group.replicas × scale。
	// 例如 scale=0.5 表示副本数减半，scale=2 表示资源翻倍。
	// 支持的范围为 0.0 到 1.0，其中 0.0 表示完全缩减资源。
	Scale *float32 `json:"scale,omitempty"`

	// Start 时段起始时间（24小时制，HH:mm）
	Start *string `json:"start,omitempty"`
}

// SubmitMatchingIntentParams defines parameters for SubmitMatchingIntent.
type SubmitMatchingIntentParams struct {
	// LastDecisionTime 已废弃，后续使用decisionDeadline。保留此字段，做向前兼容已废弃
	LastDecisionTime *int64  `form:"lastDecisionTime,omitempty" json:"lastDecisionTime,omitempty"`
	IdempotencyKey   *string `json:"Idempotency-Key,omitempty"`
}

// CancelMatchingIntentJSONBody defines parameters for CancelMatchingIntent.
type CancelMatchingIntentJSONBody struct {
	// Reason 取消原因（可选）
	Reason *string `json:"reason,omitempty"`
}

// CommitMatchingIntentJSONBody defines parameters for CommitMatchingIntent.
type CommitMatchingIntentJSONBody struct {
	// Reason 提交原因
	Reason *string `json:"reason,omitempty"`
}

// GetMatchingDetailParams defines parameters for GetMatchingDetail.
type GetMatchingDetailParams struct {
	// WithDebugInfo 是否包括详细的 debugInfo
	WithDebugInfo *bool `form:"withDebugInfo,omitempty" json:"withDebugInfo,omitempty"`
}

// SubmitMatchingIntentJSONRequestBody defines body for SubmitMatchingIntent for application/json ContentType.
type SubmitMatchingIntentJSONRequestBody = MatchingIntent

// ValidateMatchingIntentJSONRequestBody defines body for ValidateMatchingIntent for application/json ContentType.
type ValidateMatchingIntentJSONRequestBody = MatchingIntent

// CancelMatchingIntentJSONRequestBody defines body for CancelMatchingIntent for application/json ContentType.
type CancelMatchingIntentJSONRequestBody CancelMatchingIntentJSONBody

// CommitMatchingIntentJSONRequestBody defines body for CommitMatchingIntent for application/json ContentType.
type CommitMatchingIntentJSONRequestBody CommitMatchingIntentJSONBody
