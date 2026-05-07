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

import (
	"strconv"
	"strings"
)

func SplitStrToList(s, sep string) []string {
	if len(s) == 0 {
		return nil
	}
	return strings.Split(s, sep)
}

func JoinInt64ListToStr(i64List []int64, sep string) string {
	var strList []string
	for _, i64 := range i64List {
		strList = append(strList, strconv.FormatInt(i64, 10))
	}
	return strings.Join(strList, sep)
}

func SplitStrPtrToList(s *string, sep string) []string {
	if s == nil {
		return nil
	}
	return SplitStrToList(*s, sep)
}

func JoinErrorListToString(errList []error, sep string) string {
	var strList []string
	for _, err := range errList {
		if err == nil {
			continue
		}
		strList = append(strList, err.Error())
	}
	return strings.Join(strList, sep)
}

func TruncateString(str string, maxLength int) string {
	if len(str) <= maxLength {
		return str
	}
	return str[:maxLength]
}
