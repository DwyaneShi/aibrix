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

package tcemock

import (
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/catalog"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provisioner"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/types"
	"github.com/vllm-project/aibrix/apps/console/api/store"
)

// init self-registers the in-process TCE mock provider. It pairs with the
// planner's tceMockBackend to let the console boot end-to-end in demo
// deployments where the real TCE control plane is unreachable. The mock
// holds no external credentials, so registration always succeeds.
func init() {
	provisioner.Register(types.ResourceProvisionTypeTCEMock, func(s store.Store) (provisioner.Provisioner, error) {
		return newProvisioner(s)
	})
	catalog.Register(types.ResourceProvisionTypeTCEMock, func() (catalog.Catalog, error) {
		return newCatalog()
	})
}
