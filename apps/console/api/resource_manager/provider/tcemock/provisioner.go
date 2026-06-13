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

// In-process mock TCE provisioner. Used in demo deployments where the real
// TCE control plane is unreachable. Provision returns immediately with
// status=Running so the planner's waitForProvisionReady polls succeed on
// the first iteration and CreateBatch fires.

package tcemock

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provisioner"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
	"github.com/vllm-project/aibrix/apps/console/api/store"
)

// tceMockProvisioner satisfies provisioner.Provisioner with hard-coded
// "always running" behavior. Persists results through the same store the
// real provisioners use so List, restart-recovery, and the lazy-sync paths
// in the planner work unchanged.
type tceMockProvisioner struct {
	store store.Store
}

// newProvisioner creates a new TCE mock provisioner.
func newProvisioner(s store.Store) (provisioner.Provisioner, error) {
	return &tceMockProvisioner{store: s}, nil
}

// Type returns the provisioner type.
func (p *tceMockProvisioner) Type() types.ResourceProvisionType {
	return types.ResourceProvisionTypeTCEMock
}

// Provision returns a synthetic "running" ProvisionResult immediately.
// Idempotent on IdempotencyKey via the store, mirroring the real TCE
// and K8s provisioners.
func (p *tceMockProvisioner) Provision(ctx context.Context, req *types.ResourceProvision) (*types.ProvisionResult, error) {
	if req == nil {
		return nil, types.ErrInvalidArgs
	}
	if req.IdempotencyKey == "" {
		return nil, types.ErrInvalidArgs
	}

	existing, err := p.store.GetProvisionByIdempotencyKey(ctx, req.IdempotencyKey)
	if err == nil && existing != nil {
		return existing, nil
	}
	if err != nil && status.Code(err) != codes.NotFound {
		return nil, fmt.Errorf("get provision by idempotency key: %w", err)
	}

	now := time.Now().UTC()
	result := &types.ProvisionResult{
		ProvisionID:    uuid.New().String(),
		IdempotencyKey: req.IdempotencyKey,
		Provider:       string(p.Type()),
		Status:         types.ProvisionStatusRunning,
		CreatedAt:      now,
		UpdatedAt:      now,
		// Empty TCEProvisionDetail is sufficient: the demo backend
		// (tceMockBackend) ignores the provision payload and stamps
		// hardcoded cluster/pool fields onto the planner decision.
		ExtensionProvisionResultDetails: types.ExtensionProvisionResultDetails{
			TCE: &types.TCEProvisionDetail{},
		},
	}

	if err := p.store.UpsertProvision(ctx, result); err != nil {
		return nil, fmt.Errorf("upsert provision: %w", err)
	}
	klog.Infof("[tce-mock] provision created provision_id=%q idempotency_key=%q",
		result.ProvisionID, result.IdempotencyKey)
	return result, nil
}

// Release marks the provision as released. No external resources to free.
func (p *tceMockProvisioner) Release(ctx context.Context, provisionID string) error {
	provision, err := p.store.GetProvision(ctx, provisionID)
	if err != nil {
		return fmt.Errorf("get provision: %w", err)
	}
	if provision == nil {
		return types.ErrProvisionNotFound
	}
	if provision.Status == types.ProvisionStatusReleased {
		return nil
	}
	return p.store.UpdateProvisionStatus(ctx, provisionID, types.ProvisionStatusReleased)
}

// List retrieves provisions matching the given criteria.
func (p *tceMockProvisioner) List(ctx context.Context, opts *types.ListOptions) ([]*types.ProvisionResult, error) {
	if opts == nil {
		opts = &types.ListOptions{}
	}
	return p.store.ListProvisions(ctx, opts)
}
