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

package utils

import "golang.org/x/exp/constraints"

func SliceContain[T comparable](slice []T, element T) bool {
	for _, ele := range slice {
		if ele == element {
			return true
		}
	}
	return false
}

func SliceConvert[T1, T2 constraints.Integer](slice1 []T1) (slice2 []T2) {
	for _, ele1 := range slice1 {
		slice2 = append(slice2, T2(ele1))
	}
	return slice2
}

func SliceFindNext[T comparable](slice []T, current T) (isFind bool, next T) {
	for idx, element := range slice {
		if idx <= len(slice)-2 && element == current {
			return true, slice[idx+1]
		}
	}
	return false, current
}

func SliceExcept[T comparable](slice1, slice2 []T) (slice3 []T) {
	for _, ele1 := range slice1 {
		if !SliceContain(slice2, ele1) {
			slice3 = append(slice3, ele1)
		}
	}
	return slice3
}

func SliceUnique[T comparable](slice1 []T) (slice2 []T) {
	slice2 = make([]T, 0, len(slice1))
	for _, ele1 := range slice1 {
		if !SliceContain(slice2, ele1) {
			slice2 = append(slice2, ele1)
		}
	}
	return slice2
}

func SplitSlice[T any](slice []T, size int) (slices [][]T) {
	if len(slice) == 0 {
		return slices
	}
	if size <= 0 {
		panic("size must be greater than 0")
	}
	for i := 0; i < len(slice); i += size {
		end := i + size
		if end > len(slice) {
			end = len(slice)
		}
		slices = append(slices, slice[i:end])
	}
	return slices
}

func SliceEqual[T comparable](slice1, slice2 []T) bool {
	if len(slice1) != len(slice2) {
		return false
	}
	for _, ele1 := range slice1 {
		if !SliceContain(slice2, ele1) {
			return false
		}
	}
	return true
}
