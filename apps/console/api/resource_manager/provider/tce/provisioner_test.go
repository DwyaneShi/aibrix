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
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/config"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/resource_manager_client/scheduled_plan_types"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/utils"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
	"github.com/vllm-project/aibrix/apps/console/api/store"
)

func TestTCEProvisioner_ProvisionListRelease(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ctx := context.Background()
	s := store.NewMemoryStore(nil)

	mockRMClient := resource_manager_client.NewMockClient(ctrl)
	mockBQClient := bytequota_client.NewMockClient(ctrl)

	clientset := &tceClientset{
		RegionConfig: &config.RegionConfig{
			ScheduledMatchFE: "http://test-fe",
		},
		ResourceManagerClient: mockRMClient,
		ByteQuotaClient:       mockBQClient,
	}

	provisioner, err := NewTCEProvisioner(s, clientset)
	require.NoError(t, err)

	// Test data
	idempotencyKey := "test-idempotency-key-123"
	matchId := "match-123"

	// Setup mock for GetBusinessLineInfo (called during Provision)
	mockBQClient.EXPECT().GetBusinessLineInfo(gomock.Any(), "test-psm", "Compute").
		Return(&bytequota_client.BusinessLineInfo{
			BusinessLineId:   "test-business-line-id",
			BusinessLineName: "test-business-line-name",
			ResourceGroupId:  "test-resource-group-id",
		}, nil)

	// Setup mock for CreateScheduledMatch
	mockRMClient.EXPECT().CreateScheduledMatch(gomock.Any(), gomock.Any()).
		DoAndReturn(func(ctx context.Context, req *scheduled_plan_types.MatchingIntentRequest) (*scheduled_plan_types.MatchingResult, error) {
			assert.Equal(t, idempotencyKey, req.IdempotencyKey)
			return &scheduled_plan_types.MatchingResult{
				MatchId: matchId,
				Status:  scheduled_plan_types.MatchingResultStatusBooking,
			}, nil
		})

	// Call Provision
	startTime := time.Now().Add(time.Minute * 10).UTC()
	endTime := startTime.Add(time.Minute * 60).UTC()
	req := &types.ResourceProvision{
		IdempotencyKey: idempotencyKey,
		Spec: types.ResourceProvisionSpec{
			Credential: types.ResourceCredential{
				Provider: types.ResourceProvisionTypeTCE,
				ExtensionResourceCredentials: types.ExtensionResourceCredentials{
					TCE: &types.TCECredential{
						PSM: utils.ToPtr("test-psm"),
					},
				},
			},
			Groups: &[]types.ResourceGroupSpec{
				{
					GpusPerReplica: 8,
					Replicas:       utils.ToPtr(4),
				},
			},
			TimeWindow: &types.TimeWindow{
				StartTime: startTime,
				EndTime:   &endTime,
			},
		},
	}

	result, err := provisioner.Provision(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, types.ProvisionStatusProvisioning, result.Status)
	assert.Equal(t, matchId, result.TCE.MatchId)
	assert.NotEmpty(t, result.ProvisionID)
	provisionID := result.ProvisionID

	// Verify store has provision with correct status
	storedResult, err := s.GetProvision(ctx, provisionID)
	require.NoError(t, err)
	require.NotNil(t, storedResult)
	assert.Equal(t, types.ProvisionStatusProvisioning, storedResult.Status)
	initialUpdatedAt := storedResult.UpdatedAt

	// Setup mock for GetScheduledMatch - simulate provisioning to success
	// First two calls return booking, third returns success
	mockRMClient.EXPECT().GetMatchTimeline(gomock.Any(), matchId).Return([]scheduled_plan_types.MatchTimelineEntry{}, nil).AnyTimes()
	mockRMClient.EXPECT().GetScheduledMatch(gomock.Any(), matchId).
		Return(&scheduled_plan_types.MatchingResult{
			MatchId: matchId,
			Status:  scheduled_plan_types.MatchingResultStatusBooking,
		}, nil).Times(2)

	zone := "CN"
	dc := "LF"
	physicalCluster := "test-physical-cluster"
	logicalCluster := "test-logical-cluster"
	mockRMClient.EXPECT().GetScheduledMatch(gomock.Any(), matchId).
		Return(&scheduled_plan_types.MatchingResult{
			MatchId: matchId,
			Status:  scheduled_plan_types.MatchingResultStatusSuccess,
			GroupResults: &scheduled_plan_types.GroupResults{
				{
					GroupIndex: 0,
					AllocationSegments: []scheduled_plan_types.AllocationSegment{
						{
							Id:              "segment-1",
							Allocated:       true,
							AcceleratorType: "A100",
							Count:           utils.ToPtr(32),
							Replicas:        utils.ToPtr(4),
							Cluster: scheduled_plan_types.Cluster{
								Zone:            &zone,
								Dc:              &dc,
								PhysicalCluster: &physicalCluster,
								LogicalCluster:  &logicalCluster,
							},
							TimeWindow: scheduled_plan_types.TimeWindow{
								StartTime: time.Now().UTC(),
							},
						},
					},
				},
			},
		}, nil)

	// Poll List until status becomes Running
	maxPolls := 10
	for i := 0; i < maxPolls; i++ {
		results, err := provisioner.List(ctx, nil)
		require.NoError(t, err)
		require.Len(t, results, 1)

		if results[0].Status == types.ProvisionStatusRunning {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Verify status changed to Running
	storedResult, err = s.GetProvision(ctx, provisionID)
	require.NoError(t, err)
	assert.Equal(t, types.ProvisionStatusRunning, storedResult.Status)
	assert.True(t, storedResult.UpdatedAt.After(initialUpdatedAt),
		"UpdatedAt should advance when the provision changes status")

	// Setup mock for CancelScheduledMatch
	mockRMClient.EXPECT().CancelScheduledMatch(gomock.Any(), matchId).
		Return(&scheduled_plan_types.MatchingResult{
			MatchId: matchId,
			Status:  scheduled_plan_types.MatchingResultStatusCancelled,
		}, nil)

	// Call Release
	err = provisioner.Release(ctx, provisionID)
	require.NoError(t, err)

	// Verify store has provision with Released status
	storedResult, err = s.GetProvision(ctx, provisionID)
	require.NoError(t, err)
	assert.Equal(t, types.ProvisionStatusReleased, storedResult.Status)
}

func TestBuildMatchingGroupsPreservesRTX6000DResources(t *testing.T) {
	cpuCores := 28
	preferredTypes := []string{"NVIDIA-RTX-6000D"}
	groups := []types.ResourceGroupSpec{
		{
			GpusPerReplica:     2,
			Replicas:           utils.ToPtr(3),
			CpuCoresPerReplica: &cpuCores,
			AcceleratorPreference: &types.AcceleratorPreference{
				PreferredTypes: &preferredTypes,
			},
		},
	}

	matchingGroups, err := buildMatchingGroups(&groups)
	require.NoError(t, err)
	require.NotNil(t, matchingGroups)
	require.Len(t, *matchingGroups, 1)

	group := (*matchingGroups)[0]
	require.NotNil(t, group.CpuCores)
	assert.Equal(t, 28, *group.CpuCores)
	require.NotNil(t, group.AcceleratorPreference.PreferredTypes)
	assert.Equal(t, []string{"NVIDIA-RTX-6000D"}, *group.AcceleratorPreference.PreferredTypes)
}

func TestToMatchingTimeWindowAdjustsDurationWhenTruncatedStartIsPast(t *testing.T) {
	now := time.Date(2026, time.August, 19, 10, 30, 0, 0, time.UTC)
	startTime := now.Add(5 * time.Minute)
	endTime := startTime.Add(6 * time.Hour)
	durationHours := 1

	got, err := toMatchingTimeWindow(&types.TimeWindow{
		StartTime:   startTime,
		EndTime:     &endTime,
		MinDuration: &durationHours,
		MaxDuration: &durationHours,
	}, now)

	require.NoError(t, err)
	require.NotNil(t, got)
	require.NotNil(t, got.EndTime)
	assert.Equal(t, time.Date(2026, time.August, 19, 10, 0, 0, 0, time.UTC), got.StartTime)
	assert.Equal(t, time.Date(2026, time.August, 19, 16, 0, 0, 0, time.UTC), *got.EndTime)
	require.NotNil(t, got.FlexibleAllocation)
	require.NotNil(t, got.FlexibleAllocation.MinDuration)
	require.NotNil(t, got.FlexibleAllocation.MaxDuration)
	assert.Equal(t, 1, *got.FlexibleAllocation.MinDuration)
	assert.Equal(t, 2, *got.FlexibleAllocation.MaxDuration)
}

func TestToMatchingTimeWindowKeepsDurationWhenTruncatedStartIsFuture(t *testing.T) {
	now := time.Date(2026, time.August, 19, 10, 58, 0, 0, time.UTC)
	startTime := now.Add(5 * time.Minute)
	endTime := startTime.Add(6 * time.Hour)
	durationHours := 1

	got, err := toMatchingTimeWindow(&types.TimeWindow{
		StartTime:   startTime,
		EndTime:     &endTime,
		MinDuration: &durationHours,
		MaxDuration: &durationHours,
	}, now)

	require.NoError(t, err)
	require.NotNil(t, got)
	require.NotNil(t, got.EndTime)
	assert.Equal(t, time.Date(2026, time.August, 19, 11, 0, 0, 0, time.UTC), got.StartTime)
	assert.Equal(t, time.Date(2026, time.August, 19, 17, 0, 0, 0, time.UTC), *got.EndTime)
	require.NotNil(t, got.FlexibleAllocation)
	require.NotNil(t, got.FlexibleAllocation.MinDuration)
	require.NotNil(t, got.FlexibleAllocation.MaxDuration)
	assert.Equal(t, 1, *got.FlexibleAllocation.MinDuration)
	assert.Equal(t, 1, *got.FlexibleAllocation.MaxDuration)
}

func TestToMatchingTimeWindowExpandsShortWindowForAdjustedDuration(t *testing.T) {
	now := time.Date(2026, time.August, 19, 10, 20, 0, 0, time.UTC)
	startTime := now.Add(5 * time.Minute)
	endTime := startTime.Add(time.Hour)
	durationHours := 1

	got, err := toMatchingTimeWindow(&types.TimeWindow{
		StartTime:   startTime,
		EndTime:     &endTime,
		MinDuration: &durationHours,
		MaxDuration: &durationHours,
	}, now)

	require.NoError(t, err)
	require.NotNil(t, got.EndTime)
	assert.Equal(t, 2*time.Hour, got.EndTime.Sub(got.StartTime))
	require.NotNil(t, got.FlexibleAllocation)
	require.NotNil(t, got.FlexibleAllocation.MinDuration)
	require.NotNil(t, got.FlexibleAllocation.MaxDuration)
	assert.Equal(t, 1, *got.FlexibleAllocation.MinDuration)
	assert.Equal(t, 2, *got.FlexibleAllocation.MaxDuration)
}

func TestTCEProvisioner_ProvisionFailed(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ctx := context.Background()
	s := store.NewMemoryStore(nil)

	mockRMClient := resource_manager_client.NewMockClient(ctrl)
	mockBQClient := bytequota_client.NewMockClient(ctrl)

	clientset := &tceClientset{
		RegionConfig: &config.RegionConfig{
			ScheduledMatchFE: "http://test-fe",
		},
		ResourceManagerClient: mockRMClient,
		ByteQuotaClient:       mockBQClient,
	}

	provisioner, err := NewTCEProvisioner(s, clientset)
	require.NoError(t, err)

	idempotencyKey := "test-idempotency-key-failed"
	matchId := "match-failed"

	mockBQClient.EXPECT().GetBusinessLineInfo(gomock.Any(), "test-psm", "Compute").
		Return(&bytequota_client.BusinessLineInfo{
			BusinessLineId:   "test-business-line-id",
			BusinessLineName: "test-business-line-name",
			ResourceGroupId:  "test-resource-group-id",
		}, nil)

	mockRMClient.EXPECT().CreateScheduledMatch(gomock.Any(), gomock.Any()).
		Return(&scheduled_plan_types.MatchingResult{
			MatchId: matchId,
			Status:  scheduled_plan_types.MatchingResultStatusBooking,
		}, nil)

	startTime := time.Now().Add(time.Minute * 10).UTC()
	endTime := startTime.Add(time.Minute * 60).UTC()
	req := &types.ResourceProvision{
		IdempotencyKey: idempotencyKey,
		Spec: types.ResourceProvisionSpec{
			Credential: types.ResourceCredential{
				Provider: types.ResourceProvisionTypeTCE,
				ExtensionResourceCredentials: types.ExtensionResourceCredentials{
					TCE: &types.TCECredential{
						PSM: utils.ToPtr("test-psm"),
					},
				},
			},
			Groups: &[]types.ResourceGroupSpec{
				{
					GpusPerReplica: 8,
					Replicas:       utils.ToPtr(1),
				},
			},
			TimeWindow: &types.TimeWindow{
				StartTime: startTime,
				EndTime:   &endTime,
			},
		},
	}

	result, err := provisioner.Provision(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
	provisionID := result.ProvisionID

	// Setup mock for GetScheduledMatch - simulate failure
	mockRMClient.EXPECT().GetMatchTimeline(gomock.Any(), matchId).Return([]scheduled_plan_types.MatchTimelineEntry{}, nil).AnyTimes()
	mockRMClient.EXPECT().GetScheduledMatch(gomock.Any(), matchId).
		Return(&scheduled_plan_types.MatchingResult{
			MatchId: matchId,
			Status:  scheduled_plan_types.MatchingResultStatusFailed,
		}, nil)

	// Call List to update status
	results, err := provisioner.List(ctx, nil)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Verify status changed to Failed
	storedResult, err := s.GetProvision(ctx, provisionID)
	require.NoError(t, err)
	assert.Equal(t, types.ProvisionStatusFailed, storedResult.Status)
}

func TestTCEProvisioner_ReleaseIdempotent(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	ctx := context.Background()
	s := store.NewMemoryStore(nil)

	mockRMClient := resource_manager_client.NewMockClient(ctrl)
	mockBQClient := bytequota_client.NewMockClient(ctrl)

	clientset := &tceClientset{
		RegionConfig: &config.RegionConfig{
			ScheduledMatchFE: "http://test-fe",
		},
		ResourceManagerClient: mockRMClient,
		ByteQuotaClient:       mockBQClient,
	}

	provisioner, err := NewTCEProvisioner(s, clientset)
	require.NoError(t, err)

	idempotencyKey := "test-idempotency-key-release"
	matchId := "match-release"

	mockBQClient.EXPECT().GetBusinessLineInfo(gomock.Any(), "test-psm", "Compute").
		Return(&bytequota_client.BusinessLineInfo{
			BusinessLineId:   "test-business-line-id",
			BusinessLineName: "test-business-line-name",
			ResourceGroupId:  "test-resource-group-id",
		}, nil)

	mockRMClient.EXPECT().CreateScheduledMatch(gomock.Any(), gomock.Any()).
		Return(&scheduled_plan_types.MatchingResult{
			MatchId: matchId,
			Status:  scheduled_plan_types.MatchingResultStatusBooking,
		}, nil)

	startTime := time.Now().Add(time.Minute * 10).UTC()
	endTime := startTime.Add(time.Minute * 60).UTC()
	req := &types.ResourceProvision{
		IdempotencyKey: idempotencyKey,
		Spec: types.ResourceProvisionSpec{
			Credential: types.ResourceCredential{
				Provider: types.ResourceProvisionTypeTCE,
				ExtensionResourceCredentials: types.ExtensionResourceCredentials{
					TCE: &types.TCECredential{
						PSM: utils.ToPtr("test-psm"),
					},
				},
			},
			Groups: &[]types.ResourceGroupSpec{
				{
					GpusPerReplica: 8,
					Replicas:       utils.ToPtr(1),
				},
			},
			TimeWindow: &types.TimeWindow{
				StartTime: startTime,
				EndTime:   &endTime,
			},
		},
	}

	result, err := provisioner.Provision(ctx, req)
	require.NoError(t, err)
	provisionID := result.ProvisionID

	mockRMClient.EXPECT().GetMatchTimeline(gomock.Any(), matchId).Return([]scheduled_plan_types.MatchTimelineEntry{}, nil).AnyTimes()
	mockRMClient.EXPECT().CancelScheduledMatch(gomock.Any(), matchId).
		Return(&scheduled_plan_types.MatchingResult{
			MatchId: matchId,
			Status:  scheduled_plan_types.MatchingResultStatusCancelled,
		}, nil)

	// First release
	err = provisioner.Release(ctx, provisionID)
	require.NoError(t, err)

	// Second release should be idempotent (no additional cancel call expected)
	err = provisioner.Release(ctx, provisionID)
	require.NoError(t, err)
}

// TestTCEProvisioner_RealClient tests with real TCE clients.
// This test is skipped by default. Remove t.Skip() to run manually.
func TestTCEProvisioner_RealClient(t *testing.T) {
	t.Skip("Skipping test with real client. Remove this line to run manually.")

	ctx := context.Background()
	s := store.NewMemoryStore(nil)

	clientset, err := newTCEClientset()
	require.NoError(t, err)

	provisioner, err := NewTCEProvisioner(s, clientset)
	require.NoError(t, err)

	idempotencyKey := "test-idempotency-key-real-" + time.Now().UTC().Format(time.RFC3339)

	// Call Provision with real client
	startTime := time.Now().Add(time.Minute * 10).UTC()
	endTime := startTime.Add(time.Minute * 60).UTC()
	req := &types.ResourceProvision{
		IdempotencyKey: idempotencyKey,
		Spec: types.ResourceProvisionSpec{
			Credential: types.ResourceCredential{
				Provider: types.ResourceProvisionTypeTCE,
				ExtensionResourceCredentials: types.ExtensionResourceCredentials{
					TCE: &types.TCECredential{
						PSM: utils.ToPtr("inf.aibrix.platform"),
					},
				},
			},
			Groups: &[]types.ResourceGroupSpec{
				{
					GroupRole:      utils.ToPtr("prefill"),
					GpusPerReplica: 1,
					Replicas:       utils.ToPtr(1),
					AcceleratorPreference: &types.AcceleratorPreference{
						PreferredTypes: utils.ToPtr([]string{"jiuhuashan"}),
					},
				},
			},
			TimeWindow: &types.TimeWindow{
				StartTime: startTime,
				EndTime:   &endTime,
			},
		},
	}

	result, err := provisioner.Provision(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, result)
	provisionID := result.ProvisionID
	t.Logf("Provision created: provisionID=%s, matchId=%s, status=%s",
		provisionID, result.TCE.MatchId, result.Status)

	// Register cleanup to release provision even if test fails
	t.Cleanup(func() {
		err := provisioner.Release(ctx, provisionID)
		if err != nil {
			t.Logf("Warning: failed to release provision %s: %v", provisionID, err)
			return
		}
		storedResult, err := s.GetProvision(ctx, provisionID)
		if err == nil && storedResult != nil {
			assert.Equal(t, types.ProvisionStatusReleased, storedResult.Status)
		}
		t.Logf("Provision released: provisionID=%s", provisionID)
	})

	// Poll List until status becomes Running or timeout
	maxPolls := 60
	for i := 0; i < maxPolls; i++ {
		results, err := provisioner.List(ctx, nil)
		require.NoError(t, err)
		require.Len(t, results, 1)

		t.Logf("Poll %d: status=%s", i+1, results[0].Status)

		if results[0].Status == types.ProvisionStatusRunning ||
			results[0].Status == types.ProvisionStatusFailed {
			break
		}
		time.Sleep(5 * time.Second)
	}

	// Verify final status
	storedResult, err := s.GetProvision(ctx, provisionID)
	require.NoError(t, err)
	require.NotNil(t, storedResult.TCE.GroupResults)
	groupResults, err := json.Marshal(storedResult.TCE.GroupResults)
	require.NoError(t, err)
	t.Logf("Final status: %s, groupResults=%s", storedResult.Status, groupResults)
}
