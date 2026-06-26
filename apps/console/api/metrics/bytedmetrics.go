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

package metrics

import (
	"fmt"

	metricsv4 "code.byted.org/gopkg/metrics/v4"
	"code.byted.org/gopkg/metrics/v4/compatible"
)

// reservedV4GlobalTags lists names auto-injected by v4 SetTceTags.
// We drop these from user-supplied tags to avoid cardinality explosions.
var reservedV4GlobalTags = map[string]struct{}{
	"cluster": {}, "pod_name": {}, "_psm": {},
	"deploy_stage": {}, "host_v6": {}, "env_type": {},
}

// BytedSink emits metrics to the internal TSDB via
// code.byted.org/gopkg/metrics/v4 (metrics 2.0 compatible client).
type BytedSink struct {
	cli    compatible.Client
	prefix string
}

// NewBytedSink creates a sink that emits to TSDB.
// prefix is typically "{component}.{env}".
func NewBytedSink(prefix string) (*BytedSink, error) {
	cli, err := compatible.NewClient(prefix, true)
	if err != nil {
		return nil, fmt.Errorf("byted metrics init: %w", err)
	}
	return &BytedSink{cli: cli, prefix: prefix}, nil
}

func (b *BytedSink) Counter(name string, val float32, tags ...Tag) {
	_ = b.cli.EmitCounter(name, val, toV4Tags(tags)...)
}

func (b *BytedSink) Gauge(name string, val float32, tags ...Tag) {
	_ = b.cli.EmitStore(name, val, toV4Tags(tags)...)
}

func (b *BytedSink) Timer(name string, val float32, tags ...Tag) {
	_ = b.cli.EmitTimer(name, val, toV4Tags(tags)...)
}

func (b *BytedSink) Store(name string, val float32, tags ...Tag) {
	_ = b.cli.EmitStore(name, val, toV4Tags(tags)...)
}

func (b *BytedSink) Rate(name string, val float32, tags ...Tag) {
	_ = b.cli.EmitRateCounter(name, val, toV4Tags(tags)...)
}

func (b *BytedSink) Close() error { return nil }

// toV4Tags drops reserved global tags and deduplicates (last wins).
func toV4Tags(tags []Tag) []metricsv4.T {
	if len(tags) == 0 {
		return nil
	}
	out := make([]metricsv4.T, 0, len(tags))
	idx := make(map[string]int, len(tags))
	for _, t := range tags {
		if _, reserved := reservedV4GlobalTags[t.Name]; reserved {
			continue
		}
		if i, ok := idx[t.Name]; ok {
			out[i].Value = t.Value
			continue
		}
		idx[t.Name] = len(out)
		out = append(out, metricsv4.Tag(t.Name, t.Value))
	}
	return out
}
