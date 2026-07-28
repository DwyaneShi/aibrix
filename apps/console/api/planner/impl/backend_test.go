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
	"fmt"
	"reflect"
	"testing"

	plannerapi "github.com/vllm-project/aibrix/apps/console/api/planner/api"
	plannerclient "github.com/vllm-project/aibrix/apps/console/api/planner/client"
	rmtypes "github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
	"k8s.io/utils/ptr"
)

func TestDecodeEngineFromTemplate(t *testing.T) {
	// Keys are snake_case: the handler marshals the spec with
	// protojson UseProtoNames=true (engine.serve_args, not serveArgs).
	ref := &plannerapi.ModelTemplateRef{
		Spec: []byte(`{
			"engine": {
				"type": "vllm",
				"image": "vllm/vllm-openai:v0.8.5",
				"serve_args": ["--max-model-len", "4096"]
			},
			"accelerator": {"type": "A10", "count": 1}
		}`),
	}
	image, serveArgs, err := decodeEngineFromTemplate(ref)
	if err != nil {
		t.Fatalf("decodeEngineFromTemplate: %v", err)
	}
	if image != "vllm/vllm-openai:v0.8.5" {
		t.Errorf("image = %q, want vllm/vllm-openai:v0.8.5", image)
	}
	if !reflect.DeepEqual(serveArgs, []string{"--max-model-len", "4096"}) {
		t.Errorf("serveArgs = %v, want [--max-model-len 4096]", serveArgs)
	}
}

func TestDecodeEngineFromTemplateEmpty(t *testing.T) {
	image, serveArgs, err := decodeEngineFromTemplate(nil)
	if err != nil || image != "" || serveArgs != nil {
		t.Errorf("got (%q, %v, %v), want empty", image, serveArgs, err)
	}
}

func TestDefaultPlannerBackendScheduleUsesRequestedReplicas(t *testing.T) {
	backend := &defaultPlannerBackend{provider: rmtypes.ResourceProvisionTypeKubernetes}
	spec, err := backend.Schedule(context.Background(), &plannerapi.EnqueueRequest{
		ResourceRequest: &plannerapi.ResourceRequest{Replicas: 4},
		ModelTemplate: &plannerapi.ModelTemplateRef{
			Spec: []byte(`{"accelerator": {"type": "A10", "count": 1}}`),
		},
	})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if spec.Groups == nil || len(*spec.Groups) != 1 {
		t.Fatalf("groups = %#v, want one group", spec.Groups)
	}
	got := (*spec.Groups)[0].Replicas
	if got == nil || *got != 4 {
		t.Fatalf("replicas = %v, want 4", got)
	}
}

func TestTCEPlannerBackendScheduleCanonicalizesLegacyA100Types(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "canonicalize NVIDIA-prefixed type",
			input:    "NVIDIA-A100-SXM4-80GB",
			expected: "A100-SXM-80GB",
		},
		{
			name:     "canonicalize vendor-stripped legacy type",
			input:    "A100-SXM4-80GB",
			expected: "A100-SXM-80GB",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &tcePlannerBackend{}
			spec, err := backend.Schedule(context.Background(), &plannerapi.EnqueueRequest{
				ModelTemplate: &plannerapi.ModelTemplateRef{
					Spec: []byte(fmt.Sprintf(
						`{"accelerator": {"type": %q, "count": 8}}`,
						tt.input,
					)),
				},
			})
			if err != nil {
				t.Fatalf("Schedule: %v", err)
			}
			if spec.Groups == nil || len(*spec.Groups) != 1 {
				t.Fatalf("groups = %#v, want one group", spec.Groups)
			}
			preference := (*spec.Groups)[0].AcceleratorPreference
			if preference == nil || preference.PreferredTypes == nil {
				t.Fatalf("accelerator preference = %#v, want preferred types", preference)
			}
			got := *preference.PreferredTypes
			if !reflect.DeepEqual(got, []string{tt.expected}) {
				t.Fatalf("preferred types = %v, want [%s]", got, tt.expected)
			}
		})
	}
}

func TestDefaultPlannerBackendBuildResourceAllocationIncludesReplicas(t *testing.T) {
	backend := &defaultPlannerBackend{provider: rmtypes.ResourceProvisionTypeKubernetes}
	allocation := backend.BuildResourceAllocation(rmtypes.ResourceProvisionSpec{
		Groups: &[]rmtypes.ResourceGroupSpec{
			buildProvisionGroupPlan("A10", 1, 4),
		},
	}, &rmtypes.ProvisionResult{ProvisionID: "prov-1"})

	defaultAllocation, ok := allocation.(*plannerclient.DefaultResourceAllocation)
	if !ok {
		t.Fatalf("allocation type = %T, want DefaultResourceAllocation", allocation)
	}
	if defaultAllocation.ProvisionID != "prov-1" {
		t.Fatalf("provision id = %q, want prov-1", defaultAllocation.ProvisionID)
	}
	if len(defaultAllocation.ResourceDetails) != 1 {
		t.Fatalf("resource details len = %d, want 1", len(defaultAllocation.ResourceDetails))
	}
	detail := defaultAllocation.ResourceDetails[0]
	if detail.Replica != 4 {
		t.Fatalf("replica = %d, want 4", detail.Replica)
	}
	if detail.GpuType != "A10" {
		t.Fatalf("gpu type = %q, want A10", detail.GpuType)
	}
}

func TestBuildTCEResourceAllocationFallsBackToRequestedReplicas(t *testing.T) {
	count := 1
	groupResults := rmtypes.TCEGroupResults{
		{
			AllocationSegments: []rmtypes.TCEAllocationSegment{
				{
					AcceleratorType:     "NVIDIA-L20",
					AcceleratorCategory: "gpu",
					Count:               &count,
				},
			},
		},
	}
	spec := rmtypes.ResourceProvisionSpec{
		Groups: &[]rmtypes.ResourceGroupSpec{
			{Replicas: ptr.To(4)},
		},
	}
	allocation := buildTCEResourceAllocation(&rmtypes.ProvisionResult{
		ProvisionID: "prov-1",
		ExtensionProvisionResultDetails: rmtypes.ExtensionProvisionResultDetails{
			TCE: &rmtypes.TCEProvisionDetail{
				GroupResults: &groupResults,
			},
		},
	}, replicasFromProvisionSpec(spec))

	if len(allocation.ResourceDetails) != 1 {
		t.Fatalf("resource details len = %d, want 1", len(allocation.ResourceDetails))
	}
	detail := allocation.ResourceDetails[0]
	if detail.Replica != 4 {
		t.Fatalf("outer replica = %d, want 4", detail.Replica)
	}
	if len(detail.Resources) != 1 {
		t.Fatalf("resources len = %d, want 1", len(detail.Resources))
	}
	if detail.Resources[0].Replica != 4 {
		t.Fatalf("inner replica = %d, want 4", detail.Resources[0].Replica)
	}
}

func TestDefaultPlannerBackendBuildRuntimeUsesTemplateEngine(t *testing.T) {
	backend := &defaultPlannerBackend{provider: rmtypes.ResourceProvisionTypeLambdaCloud}
	rt, err := backend.BuildRuntime(&plannerapi.EnqueueRequest{
		Model: "Qwen/Qwen3-32B",
		ModelTemplate: &plannerapi.ModelTemplateRef{
			Spec: []byte(`{
				"engine": {
					"image": "vllm/vllm-openai:v0.8.5",
					"serve_args": ["--max-model-len", "4096"]
				}
			}`),
		},
	}, &rmtypes.ProvisionResult{
		LambdaCloud: &rmtypes.LambdaCloudProvisionDetail{
			GroupResults: []rmtypes.LambdaCloudGroupResult{{
				Instances: []rmtypes.LambdaCloudInstanceDetail{{
					PublicIp:    ptr.To("203.0.113.10"),
					SshPort:     ptr.To(22),
					SshUser:     "ubuntu",
					HttpBaseUrl: "http://203.0.113.10:8000",
				}},
			}},
		},
	})
	if err != nil {
		t.Fatalf("BuildRuntime: %v", err)
	}
	if rt.Target != "LambdaCloud" {
		t.Fatalf("target = %q, want LambdaCloud", rt.Target)
	}
	if got := rt.Options["image"]; got != "vllm/vllm-openai:v0.8.5" {
		t.Errorf("image option = %v, want vllm/vllm-openai:v0.8.5", got)
	}
	if got := rt.Options["vllm_args"]; !reflect.DeepEqual(got, []string{"--max-model-len", "4096"}) {
		t.Errorf("vllm_args option = %v, want [--max-model-len 4096]", got)
	}
}
