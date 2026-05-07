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

package quota_types

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/shopspring/decimal"
	"github.com/vllm-project/aibrix/apps/console/api/resource_manager/provider/tce/bytequota_client/resource_pool_types"
)

type Quota struct {
	Cpu     decimal.Decimal `json:"cpu,omitempty"`     // cpu cores
	Mem     decimal.Decimal `json:"mem,omitempty"`     // mem GBs
	Socket  decimal.Decimal `json:"socket,omitempty"`  // socket counts
	Gpu     CardResource    `json:"gpu,omitempty"`     // gpu cards
	Xpu     CardResource    `json:"xpu,omitempty"`     // xpu cards
	Npu     CardResource    `json:"npu,omitempty"`     // npu cards
	Spec    string          `json:"spec,omitempty"`    // 套餐规格，仅有状态socket服务需要
	SpecNum decimal.Decimal `json:"specNum,omitempty"` // 由套餐规格决定的socket数目，仅有状态socket服务需要
}

type CardResource map[string]decimal.Decimal

func NewEmptyQuota() *Quota {
	return &Quota{}
}

func NewQuota() *Quota {
	return &Quota{
		Cpu:    decimal.Zero,
		Mem:    decimal.Zero,
		Socket: decimal.Zero,
		Gpu:    NewCardResource(),
		Xpu:    NewCardResource(),
		Npu:    NewCardResource(),
	}
}

func NewQuotaFromInt(cpu, mem, socket int64) *Quota {
	return &Quota{
		Cpu:    decimal.NewFromInt(cpu),
		Mem:    decimal.NewFromInt(mem),
		Socket: decimal.NewFromInt(socket),
		Gpu:    NewCardResource(),
	}
}

func (q *Quota) ToString() string {
	return fmt.Sprintf("cpu:%s, mem:%s, socket:%s", q.Cpu.String(), q.Mem.String(), q.Socket.String())
}

func (q *Quota) Scan(value interface{}) error {
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New(fmt.Sprint("Failed to unmarshal JSONB value:", value))
	}
	err := json.Unmarshal(bytes, q)
	return err
}

func (q Quota) GetPart(isSocketOnly bool) *Quota {
	if isSocketOnly {
		return &Quota{
			Cpu:    decimal.Zero,
			Mem:    decimal.Zero,
			Socket: q.Socket,
			Gpu:    q.Gpu.Copy(),
			Npu:    q.Npu.Copy(),
			Xpu:    q.Xpu.Copy(),
		}
	} else {
		return &Quota{
			Cpu:    q.Cpu,
			Mem:    q.Mem,
			Socket: decimal.Zero,
			Gpu:    q.Gpu.Copy(),
			Npu:    q.Npu.Copy(),
			Xpu:    q.Xpu.Copy(),
		}
	}
}

func (q Quota) Add(q1 *Quota) *Quota {
	return &Quota{
		Cpu:    q.Cpu.Add(q1.Cpu),
		Mem:    q.Mem.Add(q1.Mem),
		Socket: q.Socket.Add(q1.Socket),
		Gpu:    q.Gpu.Add(q1.Gpu),
		Xpu:    q.Xpu.Add(q1.Xpu),
		Npu:    q.Npu.Add(q1.Npu),
	}
}

func (q Quota) Reduce(q1 *Quota) *Quota {
	return &Quota{
		Cpu:    q.Cpu.Sub(q1.Cpu),
		Mem:    q.Mem.Sub(q1.Mem),
		Socket: q.Socket.Sub(q1.Socket),
		Gpu:    q.Gpu.Reduce(q1.Gpu),
		Xpu:    q.Xpu.Reduce(q1.Xpu),
		Npu:    q.Npu.Reduce(q1.Npu),
	}
}

func (q Quota) Equal(q1 *Quota) bool {
	return q.Reduce(q1).IsZeroValue()
}

func (q Quota) Multiply(n decimal.Decimal) *Quota {
	return &Quota{
		Cpu:    q.Cpu.Mul(n),
		Mem:    q.Mem.Mul(n),
		Socket: q.Socket.Mul(n),
		Gpu:    q.Gpu.Multiply(n),
		Xpu:    q.Xpu.Multiply(n),
		Npu:    q.Npu.Multiply(n),
	}
}

func (q Quota) Div(n decimal.Decimal) *Quota {
	return &Quota{
		Cpu:    q.Cpu.Div(n),
		Mem:    q.Mem.Div(n),
		Socket: q.Socket.Div(n),
		Gpu:    q.Gpu.Div(n),
		Xpu:    q.Xpu.Div(n),
		Npu:    q.Npu.Div(n),
	}
}

func (q Quota) GetCopy() *Quota {
	return q.Multiply(decimal.NewFromFloat(1.0))
}

func (q Quota) HasNegativeValue() bool {
	return q.Cpu.IsNegative() || q.Mem.IsNegative() || q.Socket.IsNegative() || q.Gpu.HasNegativeValue() || q.Xpu.HasNegativeValue() || q.Npu.HasNegativeValue()
}

func (q Quota) HasPositiveValue() bool {
	return q.Cpu.IsPositive() || q.Mem.IsPositive() || q.Socket.IsPositive() || q.Gpu.HasPositiveValue() || q.Xpu.HasPositiveValue() || q.Npu.HasPositiveValue()
}

func (q Quota) IsZeroValue() bool {
	return q.Cpu.IsZero() && q.Mem.IsZero() && q.Socket.IsZero() && q.Gpu.IsZeroValue() && q.Xpu.IsZeroValue() && q.Npu.IsZeroValue()
}

func (q Quota) Negate() *Quota {
	res := &Quota{
		Cpu:    q.Cpu.Neg(),
		Mem:    q.Mem.Neg(),
		Socket: q.Socket.Neg(),
		Gpu:    q.Gpu.Negate(),
		Xpu:    q.Xpu.Negate(),
		Npu:    q.Npu.Negate(),
	}

	// avoid float -0 case
	if res.Cpu.IsZero() {
		res.Cpu = decimal.Zero
	}
	if res.Mem.IsZero() {
		res.Mem = decimal.Zero
	}
	if res.Socket.IsZero() {
		res.Socket = decimal.Zero
	}
	return res
}

func (q Quota) GetNegativePart() *Quota {
	negativePart := &Quota{}
	if q.Cpu.IsNegative() {
		// avoid float -0 case
		negativePart.Cpu = q.Cpu
	}
	if q.Mem.IsNegative() {
		// avoid float -0 case
		negativePart.Mem = q.Mem
	}
	if q.Socket.IsNegative() {
		// avoid float -0 case
		negativePart.Socket = q.Socket
	}
	if q.Gpu.HasNegativeValue() {
		negativePart.Gpu = q.Gpu.GetNegativePart()
	}
	if q.Npu.HasNegativeValue() {
		negativePart.Npu = q.Npu.GetNegativePart()
	}
	if q.Xpu.HasNegativeValue() {
		negativePart.Xpu = q.Xpu.GetNegativePart()
	}
	return negativePart
}

func (q Quota) GetPositivePart() *Quota {
	positivePart := &Quota{}
	if q.Cpu.IsPositive() {
		// avoid float -0 case
		positivePart.Cpu = q.Cpu
	}
	if q.Mem.IsPositive() {
		// avoid float -0 case
		positivePart.Mem = q.Mem
	}
	if q.Socket.IsPositive() {
		// avoid float -0 case
		positivePart.Socket = q.Socket
	}
	if q.Gpu.HasPositiveValue() {
		positivePart.Gpu = q.Gpu.GetPositivePart()
	}
	if q.Npu.HasPositiveValue() {
		positivePart.Npu = q.Npu.GetPositivePart()
	}
	if q.Xpu.HasPositiveValue() {
		positivePart.Xpu = q.Xpu.GetPositivePart()
	}
	return positivePart
}

func (q *Quota) InsertGpu(gpuType string, gpuNum decimal.Decimal) {
	if q.Gpu == nil {
		q.Gpu = NewCardResource()
	}
	q.Gpu[gpuType] = gpuNum
}

func (q *Quota) InsertNpu(npuType string, npuNum decimal.Decimal) {
	if q.Npu == nil {
		q.Npu = NewCardResource()
	}
	q.Npu[npuType] = npuNum
}

func (q *Quota) InsertXpu(xpuType string, xpuNum decimal.Decimal) {
	if q.Xpu == nil {
		q.Xpu = NewCardResource()
	}
	q.Xpu[xpuType] = xpuNum
}

func (q *Quota) InsertHardware(_type, kind string, num decimal.Decimal) {
	switch strings.ToLower(_type) {
	case "gpu":
		q.InsertGpu(kind, num)
	case "npu":
		q.InsertNpu(kind, num)
	case "xpu":
		q.InsertXpu(kind, num)
	}
}

func (q Quota) GetMaxTimes(isSocketOnly bool, unit *Quota) (_ decimal.Decimal, exist bool) {
	var timesList []decimal.Decimal
	if !isSocketOnly {
		if unit.Cpu.IsPositive() {
			timesList = append(timesList, decimal.Max(decimal.Zero, q.Cpu.Div(unit.Cpu)))
		}
		if unit.Mem.IsPositive() {
			timesList = append(timesList, decimal.Max(decimal.Zero, q.Mem.Div(unit.Mem)))
		}
	} else {
		if unit.Socket.IsPositive() {
			timesList = append(timesList, decimal.Max(decimal.Zero, q.Socket.Div(unit.Socket)))
		}
	}
	if gpuMinTimes, exist := q.Gpu.GetMaxTimes(unit.Gpu); exist {
		timesList = append(timesList, gpuMinTimes)
	}
	if len(timesList) == 0 {
		return decimal.Zero, false
	}
	return decimal.Min(timesList[0], timesList...), true
}

func (q Quota) GetMaxTimesOrZero(isSocketOnly bool, unit *Quota) decimal.Decimal {
	res, _ := q.GetMaxTimes(isSocketOnly, unit)
	return res
}

func (q Quota) MinByAvailable(isSocketOnly bool, available *Quota) *Quota {
	res := NewQuota()
	if !isSocketOnly {
		res.Cpu = decimal.Min(q.Cpu, available.Cpu)
		res.Mem = decimal.Min(q.Mem, available.Mem)
	} else {
		res.Socket = decimal.Min(q.Socket, available.Socket)
	}
	for model, num := range q.Gpu {
		res.Gpu[model] = decimal.Min(num, available.Gpu[model])
	}
	for model, num := range q.Xpu {
		res.Xpu[model] = decimal.Min(num, available.Xpu[model])
	}
	for model, num := range q.Npu {
		res.Npu[model] = decimal.Min(num, available.Npu[model])
	}
	return res
}

func NewCardResource() CardResource {
	return make(CardResource)
}

func (c CardResource) Copy() CardResource {
	res := NewCardResource()
	for model, num := range c {
		res[model] = num
	}
	return res
}

func (c CardResource) Add(c1 CardResource) CardResource {
	res := NewCardResource()
	for model := range c {
		res[model] = decimal.Zero
	}
	for model := range c1 {
		res[model] = decimal.Zero
	}
	for model := range res {
		res[model] = c[model].Add(c1[model])
	}
	return res
}

func (c CardResource) Reduce(c1 CardResource) CardResource {
	res := NewCardResource()
	for model := range c {
		res[model] = decimal.Zero
	}
	for model := range c1 {
		res[model] = decimal.Zero
	}
	for model := range res {
		res[model] = c[model].Sub(c1[model])
	}
	return res
}

func (c CardResource) Multiply(n decimal.Decimal) CardResource {
	res := NewCardResource()
	for model, num := range c {
		res[model] = n.Mul(num)
	}
	return res
}

func (c CardResource) Div(n decimal.Decimal) CardResource {
	res := NewCardResource()
	for model, num := range c {
		res[model] = n.Div(num)
	}
	return res
}

func (c CardResource) HasNegativeValue() bool {
	for _, num := range c {
		if num.IsNegative() {
			return true
		}
	}
	return false
}

func (c CardResource) HasPositiveValue() bool {
	for _, num := range c {
		if num.IsPositive() {
			return true
		}
	}
	return false
}

func (c CardResource) IsZeroValue() bool {
	for _, num := range c {
		if !num.IsZero() {
			return false
		}
	}
	return true
}

func (c CardResource) Negate() CardResource {
	res := NewCardResource()
	for model, num := range c {
		if !num.IsZero() {
			res[model] = num.Neg()
		}
	}
	return res
}

func (c CardResource) GetNegativePart() CardResource {
	negativePart := NewCardResource()
	for model, num := range c {
		if num.IsNegative() {
			// avoid float -0 case
			negativePart[model] = num
		}
	}
	return negativePart
}

func (c CardResource) GetPositivePart() CardResource {
	positivePart := NewCardResource()
	for model, num := range c {
		if num.IsPositive() {
			// avoid float -0 case
			positivePart[model] = num
		}
	}
	return positivePart
}

func (c CardResource) GetMaxTimes(unit CardResource) (_ decimal.Decimal, exist bool) {
	var timesList []decimal.Decimal
	for model, unitNum := range unit {
		totalNum := c[model]
		if unitNum.IsPositive() {
			timesList = append(timesList, decimal.Max(decimal.Zero, totalNum.Div(unitNum)))
		}
	}
	if len(timesList) == 0 {
		return decimal.Zero, false
	}
	return decimal.Min(timesList[0], timesList...), true
}

func NewResourceQuota(resourceItem resource_pool_types.ResourceItem, convertedUnit bool) *Quota {
	quota := NewQuota()
	if _, exist := resourceItem["cpu"]; exist {
		if _, exist := resourceItem["cpu"]["default"]; exist {
			quota.Cpu = decimal.NewFromInt(resourceItem["cpu"]["default"])
		}
	}
	if _, exist := resourceItem["memory"]; exist {
		if _, exist := resourceItem["memory"]["default"]; exist {
			quota.Mem = decimal.NewFromInt(resourceItem["memory"]["default"])
		}
	}
	if _, exist := resourceItem["socket"]; exist {
		if _, exist := resourceItem["socket"]["default"]; exist {
			quota.Socket = decimal.NewFromInt(resourceItem["socket"]["default"])
		}
	}
	if _, exist := resourceItem["gpu"]; exist {
		quota.Gpu = newCardResource(resourceItem["gpu"])
	}
	if _, exist := resourceItem["xpu"]; exist {
		quota.Xpu = newCardResource(resourceItem["xpu"])
	}
	if _, exist := resourceItem["npu"]; exist {
		quota.Npu = newCardResource(resourceItem["npu"])
	}
	if !convertedUnit {
		quota.Cpu = quota.Cpu.Div(decimal.NewFromInt(1000))
		quota.Mem = quota.Mem.Div(decimal.NewFromInt(1000 * 1024))
		quota.Socket = quota.Socket.Div(decimal.NewFromInt(1000))
		convertedGpuCardResource := make(CardResource)
		for model, num := range quota.Gpu {
			convertedGpuCardResource[model] = num.Div(decimal.NewFromInt(1000))
		}
		quota.Gpu = convertedGpuCardResource

		convertedXpuCardResource := make(CardResource)
		for model, num := range quota.Xpu {
			convertedXpuCardResource[model] = num.Div(decimal.NewFromInt(1000))
		}
		quota.Xpu = convertedXpuCardResource

		convertedNpuCardResource := make(CardResource)
		for model, num := range quota.Npu {
			convertedNpuCardResource[model] = num.Div(decimal.NewFromInt(1000))
		}
		quota.Npu = convertedNpuCardResource
	}
	return quota
}

func newCardResource(cardResource map[string]int64) CardResource {
	newCardResource := make(CardResource)
	for model, num := range cardResource {
		newCardResource[model] = decimal.NewFromInt(num)
	}
	return newCardResource
}
