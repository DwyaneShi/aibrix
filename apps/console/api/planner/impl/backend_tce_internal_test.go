/*
Copyright 2026 The Aibrix Team.

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

package impl

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	plannerapi "github.com/vllm-project/aibrix/apps/console/api/planner/api"
	rmtypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
)

func TestTCEPlannerBackendScheduleAcceptsCustomCompletionWindow(t *testing.T) {
	backend := &tcePlannerBackend{}
	spec, err := backend.Schedule(context.Background(), &plannerapi.EnqueueRequest{
		BatchParams: openai.BatchNewParams{
			CompletionWindow: "1h38min",
		},
		ModelTemplate: &plannerapi.ModelTemplateRef{
			Spec: []byte(`{"accelerator": {"type": "NVIDIA-H20", "count": 1}}`),
		},
	})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if spec.TimeWindow == nil || spec.TimeWindow.EndTime == nil {
		t.Fatalf("time window = %#v, want bounded resource window", spec.TimeWindow)
	}
	if got := spec.TimeWindow.EndTime.Sub(spec.TimeWindow.StartTime); got != 98*time.Minute {
		t.Fatalf("resource window = %v, want 98m", got)
	}
}

func TestTCEPlannerBackendScheduleSetsExactResourceDuration(t *testing.T) {
	backend := &tcePlannerBackend{}
	spec, err := backend.Schedule(context.Background(), &plannerapi.EnqueueRequest{
		BatchParams: openai.BatchNewParams{
			CompletionWindow: "6h",
		},
		ResourceRequest: &plannerapi.ResourceRequest{
			ProviderConfig: map[string]any{
				"duration": "1h",
			},
		},
		ModelTemplate: &plannerapi.ModelTemplateRef{
			Spec: []byte(`{"accelerator": {"type": "NVIDIA-H20", "count": 1}}`),
		},
	})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if spec.TimeWindow == nil || spec.TimeWindow.EndTime == nil {
		t.Fatalf("time window = %#v, want bounded resource window", spec.TimeWindow)
	}
	if got := spec.TimeWindow.EndTime.Sub(spec.TimeWindow.StartTime); got != 6*time.Hour {
		t.Fatalf("matching window = %v, want 6h", got)
	}
	if spec.TimeWindow.MinDuration == nil || *spec.TimeWindow.MinDuration != 1 {
		t.Fatalf("min duration = %v, want 1h", spec.TimeWindow.MinDuration)
	}
	if spec.TimeWindow.MaxDuration == nil || *spec.TimeWindow.MaxDuration != 1 {
		t.Fatalf("max duration = %v, want 1h", spec.TimeWindow.MaxDuration)
	}
}

func TestTCEPlannerBackendScheduleRejectsInvalidResourceDuration(t *testing.T) {
	backend := &tcePlannerBackend{}
	tests := []struct {
		name     string
		duration any
	}{
		{name: "non string", duration: 1},
		{name: "invalid format", duration: "best_effort"},
		{name: "fractional hour", duration: "90min"},
		{name: "longer than window", duration: "12h"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := backend.Schedule(context.Background(), &plannerapi.EnqueueRequest{
				BatchParams: openai.BatchNewParams{
					CompletionWindow: "6h",
				},
				ResourceRequest: &plannerapi.ResourceRequest{
					ProviderConfig: map[string]any{
						"duration": tt.duration,
					},
				},
				ModelTemplate: &plannerapi.ModelTemplateRef{
					Spec: []byte(`{"accelerator": {"type": "NVIDIA-H20", "count": 1}}`),
				},
			})
			if err == nil {
				t.Fatal("Schedule unexpectedly succeeded")
			}
		})
	}
}

func TestTCEPlannerBackendValidateRequestRejectsInvalidResourceDuration(t *testing.T) {
	backend := &tcePlannerBackend{}
	for _, duration := range []any{1, "best_effort", "90min", "12h"} {
		req := &plannerapi.EnqueueRequest{
			BatchParams: openai.BatchNewParams{CompletionWindow: "6h"},
			ResourceRequest: &plannerapi.ResourceRequest{ProviderConfig: map[string]any{
				"duration": duration,
			}},
			ModelTemplate: &plannerapi.ModelTemplateRef{Name: "template"},
		}
		err := backend.ValidateRequest(req)
		if !errors.Is(err, plannerapi.ErrInvalidJob) {
			t.Fatalf("duration %#v error = %v, want ErrInvalidJob", duration, err)
		}
	}
}

func TestTCEPlannerBackendAllocationTimeWindowUsesSegmentIntersection(t *testing.T) {
	firstStart := time.Now().UTC().Add(time.Hour)
	secondStart := firstStart.Add(10 * time.Minute)
	firstEnd := firstStart.Add(3 * time.Hour)
	secondEnd := firstStart.Add(2 * time.Hour)
	groupResults := rmtypes.TCEGroupResults{{
		AllocationSegments: []rmtypes.TCEAllocationSegment{
			{
				Allocated: true,
				TimeWindow: rmtypes.TimeWindow{
					StartTime: firstStart,
					EndTime:   &firstEnd,
				},
			},
			{
				Allocated: true,
				TimeWindow: rmtypes.TimeWindow{
					StartTime: secondStart,
					EndTime:   &secondEnd,
				},
			},
		},
	}}
	prov := &rmtypes.ProvisionResult{
		ExtensionProvisionResultDetails: rmtypes.ExtensionProvisionResultDetails{
			TCE: &rmtypes.TCEProvisionDetail{GroupResults: &groupResults},
		},
	}

	got := (&tcePlannerBackend{}).AllocationTimeWindow(prov)

	if got == nil || !got.StartTime.Equal(secondStart) {
		t.Fatalf("start time = %#v, want latest segment start %v", got, secondStart)
	}
	if got.EndTime == nil || !got.EndTime.Equal(secondEnd) {
		t.Fatalf("end time = %#v, want earliest segment end %v", got, secondEnd)
	}
}
