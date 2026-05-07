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

package resource_manager_client

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/bytedance/sonic"
)

var (
	STATUS_INTERNAL_ERR                     = &Response{Code: -1, Error: "内部错误"}
	STATUS_SUCCESS                          = &Response{Code: 0, Error: "成功"}
	STATUS_UNKOWN_ERR                       = &Response{Code: 1, Error: "未知错误"}
	STATUS_JWT_AUTH_FAILED                  = &Response{Code: 2, Error: "jwt身份认证失败"}
	STATUS_UNLOGIN                          = &Response{Code: 3, Error: "未登陆"}
	STATUS_BAD_PARAM                        = &Response{Code: 4, Error: "参数不合法"}
	STATUS_RESOURCE_GROUP_NO_PERMISSION     = &Response{Code: 11, Error: "无资源组操作权限"}
	STATUS_NODE_NO_PERMISSION               = &Response{Code: 12, Error: "无节点操作权限"}
	STATUS_PSM_NO_PERMISSION                = &Response{Code: 13, Error: "无PSM操作权限"}
	STATUS_PAAS_CLUSTER_ID_NOT_EXIST        = &Response{Code: 14, Error: "tce cluster id不存在"}
	STATUS_GALAXY_NODE_NOT_EXIST            = &Response{Code: 15, Error: "服务树节点不存在"}
	STATUS_LARK_MSG_SEND_FAIL               = &Response{Code: 16, Error: "lark通知消息发送失败"}
	STATUS_QUERY_TCE_CLUSTER_INFO_FAILED    = &Response{Code: 17, Error: "查询tce集群信息失败"}
	STATUS_LARK_GROUP_CREATE_FAILED         = &Response{Code: 18, Error: "lark群聊创建失败"}
	STATUS_BYTEPAAS_WORKLOAD_ID_NOT_EXIST   = &Response{Code: 19, Error: "bytepaas workload id不存在"}
	STATUS_BLOCKED_BY_SCP                   = &Response{Code: 20, Error: "服务策略拦截"}
	STATUS_PAAS_CLUSTER_GROUP_NOT_EXIST     = &Response{Code: 21, Error: "paas cluster group不存在"}
	STATUS_LARK_GROUP_ADD_MEMBER_FAILED     = &Response{Code: 22, Error: "lark群聊添加用户失败"}
	STATUS_LARK_UPDATE_TOP_MESSAHE_FAILED   = &Response{Code: 23, Error: "lark群聊更新置顶消息失败"}
	STATUS_LARK_ROBOT_NOT_IN_CHAT_GROUP     = &Response{Code: 24, Error: "ByteQuota机器人不在群聊中，请添加群机器人ByteQuota"}
	STATUS_BYTEPAAS_DEPLOYMENT_ID_NOT_EXIST = &Response{Code: 25, Error: "bytepaas deployment id不存在"}

	// 扩缩容任务相关错误码 2xx
	STATUS_SCALE_TASK_NOT_EXISTS                    = &Response{Code: 201, Error: "扩缩容工单不存在"}
	STATUS_SCALE_TASK_CAN_NOT_CANCEL                = &Response{Code: 202, Error: "扩缩容工单当前所处状态不支持取消"}
	STATUS_SCALE_TASK_CAN_NOT_RETRY                 = &Response{Code: 203, Error: "扩缩容工单当前所处状态不支持重试"}
	STATUS_SCALE_TASK_NO_PERMISSION                 = &Response{Code: 204, Error: "无权限操作当前扩缩容工单"}
	STATUS_SCALE_TASK_CAPACITY_CHANGED              = &Response{Code: 205, Error: "待扩缩容的服务容量信息过期，请刷新页面后重试"}
	STATUS_SCALE_TASK_CAN_NOT_IGNORE                = &Response{Code: 206, Error: "扩缩容工单当前所处状态不支持忽视异常"}
	STATUS_SCALE_TASK_CAN_NOT_MODIFY_FAIL_TOLERANCE = &Response{Code: 207, Error: "扩缩容工单当前所处状态不支持修改失败容忍参数"}
	STATUS_SCALE_TASK_SCALE_SERVICE_EMPTY_ERROR     = &Response{Code: 208, Error: "没有需要扩缩容的服务"}

	// 弹性规则组错误码 3xx
	STATUS_ELASTIC_GROUP_NOT_EXIST               = &Response{Code: 301, Error: "弹性规则组不存在"}
	STATUS_ELASTIC_GROUP_CAN_NOT_UPDATE          = &Response{Code: 302, Error: "弹性规则组当前状态不支持变更，请重试异常工单，或者取消异常工单后重新发起变更"}
	STATUS_ELASTIC_GROUP_CSD_NOT_EXIST           = &Response{Code: 303, Error: "弹性规则组下对应的集群单机房不存在"}
	STATUS_ELASTIC_GROUP_CAN_NOT_DELETE          = &Response{Code: 305, Error: "弹性规则组当前状态不支持删除，请重试异常工单，或者取消异常工单后重新发起删除"}
	STATUS_ELASTIC_GROUP_CSD_DUPLICATED          = &Response{Code: 306, Error: "集群单机房已托管到当前或其它弹性规则组中，请不要重复添加"}
	STATUS_ELASTIC_GROUP_NO_PERMISSION           = &Response{Code: 307, Error: "无权限操作当前弹性规则组"}
	STATUS_ELASTIC_GROUP_NO_CREATE_PERMISSION    = &Response{Code: 314, Error: "无权限创建弹性规则组"}
	STATUS_ELASTIC_GROUP_NO_ADD_CSD_PERMISSION   = &Response{Code: 315, Error: "无权限添加TCE集群单机房或Bernard服务"}
	STATUS_ELASTIC_GROUP_NON_DESCENDANTS_PSM     = &Response{Code: 308, Error: "PSM不是弹性规则组挂树节点的后裔节点"}
	STATUS_ELASTIC_GROUP_NODE_NOT_TOP_LEVEL      = &Response{Code: 309, Error: "弹性规则组所挂节点不是一级节点"}
	STATUS_ELASTIC_UNSUPPORTED_PHYSICAL_CLUSTERS = &Response{Code: 310, Error: "弹性规则组不支持当前物理集群"}
	STATUS_ELASTIC_STATEFUL_PAAS_CLUSTER_IDS     = &Response{Code: 311, Error: "弹性规则组不支持添加有状态的paas cluster id"}
	STATUS_TCE_CSD_INVALID_GROUP_ENV             = &Response{Code: 312, Error: "服务可能已经加入到其它控制面的弹性规则组中"}
	STATUS_BYTEPAAS_WORKLOAD_INVALID_GROUP_ENV   = &Response{Code: 313, Error: "服务可能已经加入到其它控制面的弹性规则组中"}
	STATUS_BYTEPAAS_WORKLOAD_HPA_NOT_SUPPORT     = &Response{Code: 314, Error: "bytepaas worklod不支持开启HPA"}

	// 弹性变更工单错误码 4xx
	STATUS_ELASTIC_RULE_UPDATE_TICKET_NOT_EXIST      = &Response{Code: 401, Error: "弹性规则更新工单不存在"}
	STATUS_ELASTIC_RULE_UPDATE_TICKET_CAN_NOT_CANCEL = &Response{Code: 402, Error: "弹性规则更新工单当前所处状态不支持取消"}
	STATUS_ELASTIC_RULE_UPDATE_TICKET_CAN_NOT_RETRY  = &Response{Code: 403, Error: "弹性规则更新工单当前所处状态不支持重试"}
	STATUS_ELASTIC_RULE_UPDATE_TICKET_NO_PERMISSION  = &Response{Code: 404, Error: "无权限操作当前弹性规则更新工单"}

	// Campaign错误码 5xx
	STATUS_CAMPAIGN_NOT_EXIST                                      = &Response{Code: 501, Error: "活动不存在"}
	STATUS_CAMPAIGN_BUSINESS_LINE_NOT_EXIST                        = &Response{Code: 502, Error: "活动业务线不存在"}
	STATUS_CAMPAIGN_CAN_NOT_SAVE_RESOURCES                         = &Response{Code: 503, Error: "活动已结束，不支持再保存资源"}
	STATUS_CAMPAIGN_CAN_NOT_UPSERT_RESOURCES                       = &Response{Code: 504, Error: "活动已结束，不支持再申请/释放资源"}
	STATUS_CAMPAIGN_RESOURCES_DELETE_NOT_SUPPORTED                 = &Response{Code: 505, Error: "无法删除评审中/待交付/交付中/已交付的服务"}
	STATUS_CAMPAIGN_NO_CAMPAIGN_PERMISSION                         = &Response{Code: 506, Error: "无活动工单权限"}
	STATUS_CAMPAIGN_NO_BUSINESS_LINE_PERMISSION                    = &Response{Code: 507, Error: "操作人没有业务线权限"}
	STATUS_CAMPAIGN_SUPPLY_STAGE_NOT_EXIST                         = &Response{Code: 508, Error: "活动分阶段供给不存在"}
	STATUS_CAMPAIGN_HPA_NOT_CLOSED                                 = &Response{Code: 509, Error: "部分服务的自动扩缩容没有关闭，请关闭后再重试。服务详情见Lark信息。"}
	STATUS_CAMPAIGN_NO_SRE_PERMISSION                              = &Response{Code: 510, Error: "操作人不是SRE"}
	STATUS_CAMPAIGN_NOT_BUSINESS_PRINCIPAL                         = &Response{Code: 511, Error: "非核心业务接口人不支持创建活动资源申请工单"}
	STATUS_CAMPAIGN_DELIVER_RECORD_NOT_EXIST                       = &Response{Code: 512, Error: "活动交付记录不存在"}
	STATUS_CAMPAIGN_RESOURCES_IN_EXECUTION                         = &Response{Code: 513, Error: "无法对处于交付中/回收中的服务申请资源"}
	STATUS_CAMPAIGN_RESOURCES_INVALID                              = &Response{Code: 514, Error: "资源不存在或已过期，请刷新下页面然后重试"}
	STATUS_CAMPAIGN_RESOURCES_EMPTY                                = &Response{Code: 515, Error: "部分服务的资源为空"}
	STATUS_CAMPAIGN_ONE_PSM_ACROSS_MULTI_BUSINESS_LINES            = &Response{Code: 516, Error: "单PSM不允许跨多个业务线"}
	STATUS_CAMPAIGN_CAN_NOT_FINISH_CAMPAIGN                        = &Response{Code: 517, Error: "活动已结束，无需反复结束活动"}
	STATUS_CAMPAIGN_ANY_SERVICE_RESOURCE_IN_EXECUTION              = &Response{Code: 518, Error: "无法结束活动，因为仍有资源处于交付中/回收中"}
	STATUS_CAMPAIGN_REDIS_RESOURCE_NOT_EXIST                       = &Response{Code: 519, Error: "Redis资源不存在"}
	STATUS_CAMPAIGN_MYSQL_RESOURCE_NOT_EXIST                       = &Response{Code: 520, Error: "Mysql资源不存在"}
	STATUS_CAMPAIGN_PSM_OR_CSD_NUM_EXCEEDED                        = &Response{Code: 521, Error: "活动PSM数量超限"}
	STATUS_CAMPAIGN_SRE_UNAUTHORIZED_OPERATION                     = &Response{Code: 522, Error: "此操作未经SRE授权"}
	STATUS_CAMPAIGN_ABASE_V2_PLATFORM_DB_NOT_EXIST                 = &Response{Code: 523, Error: "Abase2.0库不存在"}
	STATUS_CAMPAIGN_ABASE_V2_PLATFORM_DB_EXIST                     = &Response{Code: 524, Error: "Abase2.0库已存在"}
	STATUS_CAMPAIGN_ABASE_V2_RESOURCE_NOT_EXIST                    = &Response{Code: 525, Error: "Abase2.0资源不存在"}
	STATUS_CAMPAIGN_ABASE_V1_PLATFORM_DB_EXIST                     = &Response{Code: 526, Error: "Abase1.0库已存在"}
	STATUS_CAMPAIGN_ABASE_V1_RESOURCE_NOT_EXIST                    = &Response{Code: 527, Error: "Abase1.0资源不存在"}
	STATUS_CAMPAIGN_ABASE_V1_PLATFORM_DB_NOT_EXIST                 = &Response{Code: 528, Error: "Abase1.0库不存在"}
	STATUS_CAMPAIGN_CAMPAIGN_AND_BUSINESS_LINE_NOT_MATCH           = &Response{Code: 529, Error: "活动与业务线不匹配"}
	STATUS_CAMPAIGN_BUSINESS_LINE_AND_RESOURCES_NOT_MATCH          = &Response{Code: 530, Error: "业务线与资源不匹配"}
	STATUS_CAMPAIGN_NO_CLONE_PERMISSION                            = &Response{Code: 531, Error: "活动只能由创建人克隆，其他人无克隆权限"}
	STATUS_CAMPAIGN_ABASE_V2_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REVIEW = &Response{Code: 532, Error: "Abase2.0资源处于编辑中，不支持评审"}
	STATUS_CAMPAIGN_ABASE_V2_RESOURCES_PART_NOT_EXIST              = &Response{Code: 533, Error: "部分Abase2.0资源不存在"}
	STATUS_CAMPAIGN_ABASE_V2_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REJECT = &Response{Code: 534, Error: "Abase2.0资源不在评审中，不支持驳回"}
	STATUS_CAMPAIGN_TCE_REVIEW_RECORD_NOT_EXIST                    = &Response{Code: 535, Error: "TCE资源评审记录不存在"}
	STATUS_CAMPAIGN_TCE_RESOURCE_APPLY_CHANGE_LOG_NOT_EXIST        = &Response{Code: 536, Error: "TCE资源申请最近修改信息不存在"}
	STATUS_CAMPAIGN_TCE_RESOURCE_RELEASE_CHANGE_LOG_NOT_EXIST      = &Response{Code: 537, Error: "TCE资源回收最近修改信息不存在"}
	STATUS_CAMPAIGN_NO_CAMPAIGN_POC_PERMISSION                     = &Response{Code: 538, Error: "操作人没有活动接口人权限"}
	STATUS_CAMPAIGN_TCE_APPROVAL_FREE_QUOTA_NOT_ENOUGH             = &Response{Code: 539, Error: "活动免审批可用量不足"}
	STATUS_CAMPAIGN_TCE_APPROVAL_FREE_QUOTA_TOTAL_LESS_THAN_USED   = &Response{Code: 540, Error: "活动免审批总量小于使用量"}
	STATUS_CAMPAIGN_TCE_APPROVAL_FREE_NOT_CONFIGURED               = &Response{Code: 541, Error: "未配置活动免审批"}
	STATUS_CAMPAIGN_TCE_APPROVAL_FREE_EXPIRED                      = &Response{Code: 542, Error: "活动免审批额度已过期"}
	STATUS_DELIVER_DST_CLUSTER_INVALID                             = &Response{Code: 543, Error: "交付集群非法"}
	STATUS_CAMPAIGN_ONE_QUEUE_ACROSS_MULTI_BUSINESS_LINES          = &Response{Code: 544, Error: "单Queue不允许跨多个业务线"}
	STATUS_CAMPAIGN_NO_YARN_RELEASE_APPROVAL_PERMISSION            = &Response{Code: 545, Error: "没有yarn回收审批权限，以下角色可审批：业务接口人; 活动发起人; yarn sre"}
	STATUS_CAMPAIGN_NO_NEED_APPROVE                                = &Response{Code: 546, Error: "活动不需要审核通过操作"}
	STATUS_CAMPAIGN_CAN_NOT_MODIFY_PURPOSE_AND_TIME                = &Response{Code: 547, Error: "当前状态不可再修改「活动场景」和「活动时间」，若有修改需求请联系SRE或平台管理员"}
	STATUS_CAMPAIGN_NO_SOURCE_TYPE_PERMISSION                      = &Response{Code: 548, Error: "修改活动资源来源请联系平台管理员"}
	STATUS_CAMPAIGN_TCE_APPROVAL_FREE_NO_NEED                      = &Response{Code: 549, Error: "无需使用活动免审批额度"}
	STATUS_CAMPAIGN_RESOURCES_TAG_ERROR                            = &Response{Code: 550, Error: "填写正确的活动资源标签"}

	// 执行记录错误码 10xx
	STATUS_DELIVER_RECORD_NOT_EXIST                = &Response{Code: 1001, Error: "交付计划不存在"}
	STATUS_DELIVER_RECORD_CAN_NOT_EDIT             = &Response{Code: 1002, Error: "交付计划当前状态不支持编辑"}
	STATUS_DELIVER_RECORD_CAN_NOT_CONFIRM          = &Response{Code: 1003, Error: "交付计划当前状态不支持确认"}
	STATUS_DELIVER_RECORD_CAN_NOT_START            = &Response{Code: 1004, Error: "交付计划当前状态不支持开始"}
	STATUS_DELIVER_RECORD_CAN_NOT_DELETE           = &Response{Code: 1005, Error: "交付计划当前状态不支持删除"}
	STATUS_DELIVER_RECORD_EMPTY                    = &Response{Code: 1006, Error: "交付计划的服务列表不能为空"}
	STATUS_DELIVER_RECORD_SCALE_CPU_AMOUNT_INVALID = &Response{Code: 1007, Error: "交付计划CPU扩容核数非法"}
	STATUS_DELIVER_RECORD_CAN_NOT_REDELIVER        = &Response{Code: 1008, Error: "执行记录当前状态不支持重新交付"}
	STATUS_DELIVER_RECORD_CAN_NOT_ABORT_DELIVER    = &Response{Code: 1009, Error: "执行记录当前状态不支持放弃交付"}
	STATUS_DELIVER_RECORD_TYPE_CAN_NOT_REDELIVER   = &Response{Code: 1010, Error: "（类型为回收的）执行记录不支持重新回收"}
	STATUS_DELIVER_RECORD_CAN_NOT_COPY_TO_CLONE    = &Response{Code: 1011, Error: "交付计划当前状态不支持生成克隆计划"}

	// Mutex错误码 6xx
	STATUS_MUTEX_LOCKED = &Response{Code: 601, Error: "服务处于其它流程中"}

	// Admin错误码 7XX
	STATUS_ADMIN_TICKET_NOT_EXIST                      = &Response{Code: 701, Error: "工单不存在"}
	STATUS_ADMIN_TICKET_NO_PERMISSION                  = &Response{Code: 702, Error: "操作人不是管理员"}
	STATUS_ADMIN_TICKET_STATUS_CAN_NOT_OPERATE         = &Response{Code: 703, Error: "工单已结束，无法操作"}
	STATUS_ADMIN_TICKET_RESOURCE_NOT_ENOUGH_TO_OPERATE = &Response{Code: 703, Error: "资源量不足"}

	// Campaign RMQ错误码 8XX
	STATUS_RMQ_TOPIC_NOT_EXIST                                = &Response{Code: 801, Error: "Rmq Topic不存在"}
	STATUS_RMQ_TOPIC_EXIST                                    = &Response{Code: 802, Error: "Rmq Topic已存在"}
	STATUS_CAMPAIGN_RMQ_RESOURCE_NOT_EXIST                    = &Response{Code: 803, Error: "Rmq资源不存在"}
	STATUS_CAMPAIGN_RMQ_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REJECT = &Response{Code: 804, Error: "Rmq资源不在评审中，不支持驳回"}
	STATUS_CAMPAIGN_RMQ_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REVIEW = &Response{Code: 805, Error: "Rmq资源不在评审中，不支持评审"}
	STATUS_CAMPAIGN_RMQ_RESOURCES_PART_NOT_EXIST              = &Response{Code: 906, Error: "部分Rmq资源申请不存在"}

	// Campaign BMQ错误码 9XX
	STATUS_BMQ_TOPIC_NOT_EXIST                                = &Response{Code: 901, Error: "Bmq Topic不存在"}
	STATUS_BMQ_TOPIC_EXIST                                    = &Response{Code: 902, Error: "Bmq Topic已存在"}
	STATUS_CAMPAIGN_BMQ_RESOURCE_NOT_EXIST                    = &Response{Code: 903, Error: "Bmq资源不存在"}
	STATUS_CAMPAIGN_BMQ_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REJECT = &Response{Code: 904, Error: "Bmq资源不在评审中，不支持驳回"}
	STATUS_CAMPAIGN_BMQ_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REVIEW = &Response{Code: 905, Error: "Bmq资源不在评审中，不支持评审"}
	STATUS_CAMPAIGN_BMQ_RESOURCES_PART_NOT_EXIST              = &Response{Code: 906, Error: "部分Bmq资源申请不存在"}

	// Campaign ByteGraph错误码10XX
	STATUS_BYTE_GRAPH_CLUSTER_NOT_EXIST                              = &Response{Code: 1001, Error: "ByteGraph 集群不存在"}
	STATUS_BYTE_GRAPH_CLUSTER_EXIST                                  = &Response{Code: 1002, Error: "ByteGraph 集群已存在"}
	STATUS_CAMPAIGN_BYTE_GRAPH_RESOURCE_NOT_EXIST                    = &Response{Code: 1003, Error: "ByteGraph资源申请不存在"}
	STATUS_CAMPAIGN_BYTE_GRAPH_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REJECT = &Response{Code: 1004, Error: "ByteGraph资源申请不在评审中，不支持驳回"}
	STATUS_CAMPAIGN_BYTE_GRAPH_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REVIEW = &Response{Code: 1005, Error: "ByteGraph资源申请不在评审中，不支持评审"}
	STATUS_CAMPAIGN_BYTE_GRAPH_RESOURCES_PART_NOT_EXIST              = &Response{Code: 1006, Error: "部分ByteGraph资源申请不存在"}

	// Campaign Tos错误码11XX
	STATUS_TOS_BUCKET_NOT_EXIST                               = &Response{Code: 1101, Error: "Tos Bucket不存在"}
	STATUS_TOS_BUCKET_EXIST                                   = &Response{Code: 1102, Error: "Tos Bucket已存在"}
	STATUS_CAMPAIGN_TOS_RESOURCE_NOT_EXIST                    = &Response{Code: 1103, Error: "Tos资源不存在"}
	STATUS_CAMPAIGN_TOS_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REJECT = &Response{Code: 1104, Error: "Tos资源不在评审中，不支持驳回"}
	STATUS_CAMPAIGN_TOS_RESOURCE_NOT_IN_REVIEW_CAN_NOT_REVIEW = &Response{Code: 1105, Error: "Tos资源不在评审中，不支持评审"}
	STATUS_CAMPAIGN_TOS_RESOURCES_PART_NOT_EXIST              = &Response{Code: 1106, Error: "部分Tos资源申请不存在"}
	STATUS_CAMPAIGN_TOS_BUCKET_FUZZY_SEARCH_FAILED            = &Response{Code: 1107, Error: "Tos bucket名称模糊搜索失败"}
	STATUS_CAMPAIGN_TOS_BUCKET_META_QUERY_FAILED              = &Response{Code: 1108, Error: "Tos bucket详细信息查询失败"}
	STATUS_CAMPAIGN_TOS_BUCKET_PEAK_QUERY_FAILED              = &Response{Code: 1109, Error: "Tos bucket峰值查询失败"}

	// Permission Management错误码12XX
	STATUS_PERMISSION_MANAGEMENT_NO_INF_OP_PERMISSION              = &Response{Code: 1201, Error: "没有架构运营权限"}
	STATUS_PERMISSION_MANAGEMENT_POC_NOT_EXIST                     = &Response{Code: 1202, Error: "权限名称不存在"}
	STATUS_PERMISSION_MANAGEMENT_POC_DUPLICATED                    = &Response{Code: 1203, Error: "权限名称已重复"}
	STATUS_PERMISSION_MANAGEMENT_NO_CREATE_BUSINESS_POC_PERMISSION = &Response{Code: 1204, Error: "没有新建业务接口人权限（需要架构运营或活动接口人权限）"}
	STATUS_PERMISSION_MANAGEMENT_NO_MODIFY_BUSINESS_POC_PERMISSION = &Response{Code: 1205, Error: "没有编辑业务接口人权限（需要架构运营、活动接口人权限或业务接口人权限）"}
	STATUS_PERMISSION_MANAGEMENT_NO_MODIFY_INF_POC_PERMISSION      = &Response{Code: 1206, Error: "没有编辑基础架构接口人权限（需要架构运营或基础架构接口人权限）"}

	// Calendar错误码13XX
	STATUS_CALENDAR_NOT_EXIST                            = &Response{Code: 1301, Error: "日历不存在"}
	STATUS_CALENDAR_NO_MODIFY_CALENDAR_PERMISSION        = &Response{Code: 1302, Error: "没有活动日历的修改权限"}
	STATUS_CALENDAR_NO_CREATE_INF_ONCALL_PERMISSION      = &Response{Code: 1303, Error: "没有基础架构值班表创建权限"}
	STATUS_CALENDAR_NO_CREATE_BUSINESS_ONCALL_PERMISSION = &Response{Code: 1304, Error: "没有业务值班表创建权限"}
	STATUS_CALENDAR_ONCALL_NOT_EXIST                     = &Response{Code: 1305, Error: "日历值班表不存在"}
	STATUS_CALENDAR_NO_MODIFY_ONCALL_PERMISSION          = &Response{Code: 1306, Error: "没有日历值班表修改权限"}
	STATUS_CALENDAR_OPERATOR_NOT_IN_DUTY_LIST            = &Response{Code: 1307, Error: "操作人不在当前签到或巡检任务中"}
	STATUS_CALENDAR_OPERATOR_ALREADY_CHEDKED             = &Response{Code: 1308, Error: "操作人不可重复签到或巡检"}
	STATUS_CALENDAR_NO_TASK_TO_ALERT                     = &Response{Code: 1309, Error: "没有需要提醒的签到或巡检"}

	// Campaign Topic错误码14XX
	STATUS_CAMPAIGN_TOPIC_NOT_EXIST   = &Response{Code: 1401, Error: "活动主题不存在"}
	STATUS_CAMPAIGN_HAS_RELATED_TOPIC = &Response{Code: 1402, Error: "活动已关联活动主题"}

	// Normal Goverance错误码15XX
	STATUS_CUSTOM_GROUP_NOT_EXIST                                          = &Response{Code: 1501, Error: "自定义规则组不存在"}
	STATUS_CUSTOM_GROUP_EFFECTIVE_SCOPE_INVALID                            = &Response{Code: 1502, Error: "自定义规则组生效范围不合法"}
	STATUS_CUSTOM_GROUP_USER_NO_SCALE_PERMISSION                           = &Response{Code: 1503, Error: "用户无节点或PSM的扩缩容权限"}
	STATUS_CUSTOM_GROUP_EXIST                                              = &Response{Code: 1504, Error: "规则组名称已存在"}
	STATUS_GOVERANCE_NOTICE_DAILY_TASK_NOT_EXIST                           = &Response{Code: 1505, Error: "治理提醒每日任务不存在"}
	STATUS_GOVERANCE_NOTICE_DAILY_TASK_NOT_SUPPORT_RETRY_EXIST             = &Response{Code: 1506, Error: "治理提醒每日任务当前不支持重试"}
	STATUS_PAAS_CLUSTER_OR_VDC_OF_PSM_INVALID                              = &Response{Code: 1507, Error: "PSM的集群名称或vdc不合法"}
	STATUS_PSM_RULE_MANAGED_ON_WEBCAST_DO_NOT_SUPPORT_EDIT_BY_EXCEL        = &Response{Code: 1508, Error: "PSM治理规则由ByteStable平台管理，不支持通过excel编辑"}
	STATUS_PSM_RULE_MANAGED_ON_WEBCAST_DO_NOT_SUPPORT_FIVE_VAL_MODE        = &Response{Code: 1509, Error: "PSM治理规则由ByteStable平台管理，不支持使用5水位模式"}
	STATUS_PSM_RULE_MANAGED_ON_WEBCAST_DO_NOT_SUPPORT_MULTI_DAY_GOVERNANCE = &Response{Code: 1510, Error: "PSM治理规则由ByteStable平台管理，不支持使用周级多天治理规则"}
	// Forecast Report错误码16XX
	STATUS_FORECAST_REPORT_NOT_EXIST     = &Response{Code: 1601, Error: "预估报告不存在"}
	STATUS_FORECAST_REPORT_NO_PERMISSION = &Response{Code: 1602, Error: "用户无变更权限"}
	STATUS_FORECAST_REPORT_MODEL_EXPIRED = &Response{Code: 1603, Error: "ByteBrain模型已更新，请重新预估"}

	// Ods错误码 17xx
	STATUS_ODS_SYNC_RULES_FAILED   = &Response{Code: 1701, Error: "同步服务规则到自动容量治理链路缓存失败，可能导致自动扩容不及时"}
	STATUS_ODS_RULE_NOT_CONSISTENT = &Response{Code: 1702, Error: "服务规则与自动容量治理链路缓存不一致，可能导致自动扩容不及时"}

	// Auto Scale错误码 18xx
	STATUS_AUTO_SCALE_TASK_NOT_EXIST            = &Response{Code: 1801, Error: "自动扩缩容任务不存在"}
	STATUS_AUTO_SCALE_TASKS_NO_PERMISSION       = &Response{Code: 1802, Error: "用户无以下PSM权限，无法操作自动扩缩容任务"}
	STATUS_AUTO_SCALE_TASK_NOT_SUPPORT_ROLLBACK = &Response{Code: 1803, Error: "此自动扩缩容任务不支持回滚，" +
		"可能原因：1.当前仅支持自动缩容任务回滚；2.自动扩缩容任务未完成；3.自动扩缩容任务创建时间已超过2天"}
	// Node Rule错误码 19xx
	STATUS_NODE_RULE_NO_NODES_PERMISSION = &Response{Code: 1901, Error: "无以下节点ID权限"}

	// Resource Lending错误码 20xx
	STATUS_RESOURCE_LENDING_TICKET_NOT_EXIST             = &Response{Code: 2001, Error: "拆借工单不存在"}
	STATUS_BORROW_CONFIG_NOT_EXIST                       = &Response{Code: 2002, Error: "受让方规则不存在"}
	STATUS_NO_REVIEW_PERMISSION                          = &Response{Code: 2003, Error: "无审核权限"}
	STATUS_RESOURCE_LENDING_RECORD_NOT_EXIST             = &Response{Code: 2004, Error: "拆借记录不存在"}
	STATUS_RESOURCE_LENDING_RECORD_CAN_NOT_RETRY         = &Response{Code: 2005, Error: "拆借记录非异常状态，无法重试"}
	STATUS_RESOURCE_LENDING_RECORD_CAN_NOT_START_LEND    = &Response{Code: 2006, Error: "拆借记录缺少必要配置，或者非待出让状态，无法开始出让"}
	STATUS_RESOURCE_LENDING_RECORD_CAN_NOT_START_RECYCLE = &Response{Code: 2007, Error: "拆借记录缺少必要配置，或者非待回收状态，无法开始回收"}
	STATUS_RESOURCE_LENDING_RECORD_CAN_NOT_FORCE_START   = &Response{Code: 2008, Error: "拆借记录缺少必要配置，或者非等待状态，无法跳转下一步"}

	// Scheduled Plan错误码 21xx
	STATUS_SCHEDULED_PLAN_TICKET_STATUS_NOT_TO_REVIEW = &Response{Code: 2101, Error: "工单不是待审核阶段，无法审核"}

	// Auto Scale错误码 18xx
	STATUS_VPA_TCE_SCALE_UNIT_NOT_EXIST = &Response{Code: 2201, Error: "vpa扩缩容任务不存在"}
)

func NewInternalError(errMsg string) *Response {
	respStatus := *STATUS_INTERNAL_ERR
	respStatus.Error = fmt.Sprintf("Internal Error:%s", errMsg)
	return &respStatus
}

func NewBadParamError(errMsg string) *Response {
	respStatus := *STATUS_BAD_PARAM
	respStatus.Error = fmt.Sprintf("Bad Param Error: %s", errMsg)
	return &respStatus
}

func NewPsmNoPermissionError(noPermissionPsms []string) *Response {
	respStatus := *STATUS_PSM_NO_PERMISSION
	respStatus.Error = fmt.Sprintf("以下PSM无权限: %s", strings.Join(noPermissionPsms, ","))
	return &respStatus
}

func NewElasticGroupNonDescendantPsmError(nonDescendantPsms []string) *Response {
	respStatus := *STATUS_ELASTIC_GROUP_NON_DESCENDANTS_PSM
	respStatus.Error = fmt.Sprintf("以下PSM不是弹性规则组挂树节点的后裔节点: %s", strings.Join(nonDescendantPsms, ","))
	return &respStatus
}

func NewPaasClusterIdNotExistError(nonExistPaasClusterIds []int64) *Response {
	respStatus := *STATUS_PAAS_CLUSTER_ID_NOT_EXIST
	respStatus.Error = fmt.Sprintf("以下tce cluster id不存在: %v", nonExistPaasClusterIds)
	return &respStatus
}

func NewElasticUnsupportedPhysicalClustersError(unsupportedPhysicalClusters []string) *Response {
	respStatus := *STATUS_ELASTIC_UNSUPPORTED_PHYSICAL_CLUSTERS
	respStatus.Error = fmt.Sprintf("弹性规则组不支持以下物理集群: %s", strings.Join(unsupportedPhysicalClusters, ","))
	return &respStatus
}

func NewElasticStatefulePaasClusterIdsError(statefulPaasClusterIds []int64) *Response {
	respStatus := *STATUS_ELASTIC_STATEFUL_PAAS_CLUSTER_IDS
	respStatus.Error = fmt.Sprintf("弹性规则组不支持以下有状态paas cluster id: %v", statefulPaasClusterIds)
	return &respStatus
}

func NewLarkMsgSendFailedError(errMsg string) *Response {
	respStatus := *STATUS_LARK_MSG_SEND_FAIL
	respStatus.Error = fmt.Sprintf("lark通知消息发送失败:%s", errMsg)
	return &respStatus
}

func NewQueryTceClusterInfoFailedError(errMsg string) *Response {
	respStatus := *STATUS_QUERY_TCE_CLUSTER_INFO_FAILED
	respStatus.Error = fmt.Sprintf("查询tce集群信息失败:%s", errMsg)
	return &respStatus
}

func NewLarkGroupCreateFailedError(errMsg string) *Response {
	respStatus := *STATUS_LARK_GROUP_CREATE_FAILED
	respStatus.Error = fmt.Sprintf("lark群聊创建失败:%s", errMsg)
	return &respStatus
}

func NewMutexLockedErr(errMsg string) *Response {
	respStatus := *STATUS_MUTEX_LOCKED
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewScaleTaskCapacityChanged(errMsg string) *Response {
	respStatus := *STATUS_SCALE_TASK_CAPACITY_CHANGED
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewCampaignResourcesDeleteNotSupportedError(errMsg string) *Response {
	respStatus := *STATUS_CAMPAIGN_RESOURCES_DELETE_NOT_SUPPORTED
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewCampaignResourceInExecutionError(errMsg string) *Response {
	respStatus := *STATUS_CAMPAIGN_RESOURCES_IN_EXECUTION
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewCampaignResourceEmptyError(errMsg string) *Response {
	respStatus := *STATUS_CAMPAIGN_RESOURCES_EMPTY
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewOnePsmAcrossMultiBusinessLines(errMsg string) *Response {
	respStatus := *STATUS_CAMPAIGN_ONE_PSM_ACROSS_MULTI_BUSINESS_LINES
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewDeliverRecordScaleCpuAmountInvalidError(errMsg string) *Response {
	respStatus := *STATUS_DELIVER_RECORD_SCALE_CPU_AMOUNT_INVALID
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewCampaignResourcesInvalidError(errMsg string) *Response {
	respStatus := *STATUS_CAMPAIGN_RESOURCES_INVALID
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewDeliverRecordCanNotRedeliverError(errMsg string) *Response {
	respStatus := *STATUS_DELIVER_RECORD_CAN_NOT_REDELIVER
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewDeliverRecordCanNotAbortDeliverError(errMsg string) *Response {
	respStatus := *STATUS_DELIVER_RECORD_CAN_NOT_ABORT_DELIVER
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewCampaignPsmOrCsdNumExceededError(errMsg string) *Response {
	respStatus := *STATUS_CAMPAIGN_PSM_OR_CSD_NUM_EXCEEDED
	respStatus.Error = fmt.Sprintf("%s:%s", respStatus.Error, errMsg)
	return &respStatus
}

func NewNotFoundError(errMsg string) *Response {
	return &Response{
		Code:  404,
		Error: errMsg,
	}
}

type Response struct {
	Code   int         `json:"code,omitempty"`
	Error  string      `json:"error,omitempty"`
	Result interface{} `json:"result,omitempty"`
}

func (r *Response) WriteContentType(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
}
func (r *Response) Render(w http.ResponseWriter) error {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	response, err := sonic.Marshal(r)
	if err != nil {
		return err
	}
	_, err = w.Write(response)
	return err
}
