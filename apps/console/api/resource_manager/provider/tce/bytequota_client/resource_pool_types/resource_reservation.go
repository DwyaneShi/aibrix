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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type ResourceReservation struct {
	RecordMeta
	ResourceReservationMeta
}

type ResourceReservationMeta struct {
	QosLevel                        string                            `json:"qos_level"               bson:"qos_level"`                                       // +qos_level
	ScheduledQuotaTenant            string                            `json:"scheduled_quota_tenant"  bson:"scheduled_quota_tenant"`                          // +scheduled_quota_tenant
	ScheduledQuotaCreator           string                            `json:"scheduled_quota_creator" bson:"scheduled_quota_creator"`                         // reserve to
	CreatedTime                     time.Time                         `json:"created_time"            bson:"created_time"`                                    // created time
	EnabledTime                     time.Time                         `json:"enabled_time"            bson:"enabled_time"`                                    // enabled time
	ExpiredTime                     time.Time                         `json:"expired_time"            bson:"expired_time"`                                    // expired time
	IsPermanent                     bool                              `json:"is_permanent"            bson:"is_permanent"`                                    // is this reservation permanent
	FromPsm                         string                            `json:"from_psm"                bson:"from_psm"`                                        // created by which psm
	ReserveType                     string                            `json:"reserve_type"            bson:"reserve_type"`                                    // reserve type
	ReserveTo                       string                            `json:"reserve_to"              bson:"reserve_to"`                                      // reserve to
	Description                     string                            `json:"description"             bson:"description"`                                     // description
	ResourcePoolName                string                            `json:"resource_pool_name"      bson:"resource_pool_name"`                              // resource pool name
	ResourceGroupID                 string                            `json:"resource_group_id"       bson:"resource_group_id"`                               // resource group id
	QueueName                       string                            `json:"queue_name"              bson:"queue_name"`                                      // queue name
	BusinessLine                    string                            `json:"business_line"           bson:"business_line"`                                   // business_line
	Reservation                     ResourceItem                      `json:"reservation"             bson:"reservation"`                                     // reservation quota, contains cpu, memory, socket and gpu
	Usage                           ResourceItem                      `json:"usage"                   bson:"usage"`                                           // usage quota, contains cpu, memory, socket and gpu
	ApproximatelyCPUMemoryPerSocket ResourceItem                      `json:"approximately_cpu_memory_per_socket" bson:"approximately_cpu_memory_per_socket"` // approximately cpu and memory per socket
	Name                            string                            `json:"name"                    bson:"name"`                                            // reservation name
	Platform                        string                            `json:"platform"                bson:"platform"`                                        // platform
	Creator                         string                            `json:"creator"                 bson:"creator"`                                         // BDEE/USTS employees, creator names, the creator of resource reservation
	Namespace                       string                            `json:"namespace"               bson:"namespace"`                                       // namespace
	Labels                          map[string]string                 `json:"labels"                  bson:"labels"`                                          // Labels contains the required fields
	Annotations                     map[string]string                 `json:"annotations"             bson:"annotations"`                                     // Annotations contains the optional fields
	Alert                           bool                              `json:"alert"                   bson:"alert"`                                           // need alert
	AlertReservation                ResourceItem                      `json:"alert_reservation"       bson:"alert_reservation"`                               // alert reservation, contains cpu, memory, socket and gpu
	SupplyScheduledQuota            bool                              `json:"supply_scheduled_quota"  bson:"supply_scheduled_quota"`                          // rr quota whether to participate in the scheduled quota
	ProvisionResourcePoolName       string                            `json:"provision_resource_pool_name" bson:"provision_resource_pool_name"`               // 预留目标的资源池名称
	ProvisionResourceGroupID        string                            `json:"provision_resource_group_id" bson:"provision_resource_group_id"`                 // 预留目标的资源组ID
	ProvisionQosLevel               string                            `json:"provision_qos_level" bson:"provision_qos_level"`                                 // 预留目标的 qos level
	IsProvision                     bool                              `json:"is_provision" bson:"is_provision"`                                               // is this reservation for provision
	TopologyLevel                   string                            `json:"topology_level" bson:"topology_level"`                                           // topology
	IsGangMode                      *bool                             `json:"is_gang_mode" bson:"is_gang_mode"`                                               // gang
	MatchLabels                     map[string]string                 `json:"match_labels" bson:"match_labels"`                                               // match labels for k8s
	MatchExpressions                []metav1.LabelSelectorRequirement `json:"match_expressions" bson:"match_expressions"`                                     // match expressions for k8s
	PropagatedToFedMember           bool                              `json:"propagated_to_fed_member" bson:"propagated_to_fed_member"`
	FedMemberCluster                string                            `json:"fed_member_cluster" bson:"fed_member_cluster"`
	FedMemberNodeLevel              string                            `json:"fed_member_node_level" bson:"fed_member_node_level"`
	SaleMode                        string                            `json:"sale_mode" bson:"sale_mode"`
	Scene                           Scene                             `json:"scene" bson:"scene"`                         // increase quota for ToB or ToD
	State                           string                            `json:"state" bson:"state"`                         // Fed集群提交带拓扑感知的预留后，根据此字段判断预留是否正确下发到子集群中
	SchedulingResult                string                            `json:"scheduling_result" bson:"scheduling_result"` // 在Fed集群提交带拓扑感知的预留后，如果所有子集群全部预留失败，会将此字段设置为”'{"result":"failed","reason":"NoClusterFit"}'“
	TopologyInfo                    map[string]string                 `json:"topology_info" bson:"topology_info"`         // 在Fed集群提交带拓扑感知的预留后，返回预留的详细拓扑信息，包括子集群cluster、nodeLevel、拓扑key、拓扑value
	Admins                          []string                          `json:"admins" bson:"admins"`                       // 预留管理员列表
	ReservedResource                string                            `json:"reserved_resource" bson:"reserved_resource"` // 外到内场景下，用于表示实际出借的资源量，由tideway通过qrr anno添加
	AdditionalInfo                  map[string]string                 `json:"additional_info" bson:"additional_info"`     // 对应 qrr 的 `spec.reserveDetail.additionalInfo`
	ReserveNodes                    *bool                             `json:"reserve_nodes" bson:"reserve_nodes"`         // 对应 qrr 的 `spec.resourceSelector.reserveNodes`
	OfflineNodes                    *bool                             `json:"offline_nodes" bson:"offline_nodes"`         // 对应 qrr 的 `spec.resourceSelector.offlineNodes`
	SpecExpireTimeStamp             int64                             `json:"spec_expire_timestamp" bson:"spec_expire_timestamp"`
	StatusExpireTimeStamp           int64                             `json:"status_expire_timestamp" bson:"status_expire_timestamp"`
	IsDomainReservation             bool                              `json:"is_domain_reservation" bson:"is_domain_reservation"`
	DomainReservationWebhook        string                            `json:"domain_reservation_webhook" bson:"domain_reservation_webhook"` // 集群预留场景，当资源存在超发时，通过此webhook通知业务方
}
