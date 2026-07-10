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

package gpu_center_client

import "context"

//go:generate mockgen -source=gpu_center_client.go -destination=./gpu_center_client_mock.go -package=gpu_center_client

type Client interface {
	GetTicketPriority(ctx context.Context, ticketID int64) (*TicketPriorityResult, error)
	GetOrderTimeline(ctx context.Context, matchID string) ([]OrderTimelineEntry, error)
}
