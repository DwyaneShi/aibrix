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
	"time"
)

const (
	DateTime = "2006-01-02 15:04:05"
	Date     = "2006-01-02"
)

func StringPtrToTime(s *string) (time.Time, error) {
	if s == nil {
		return time.Time{}, nil
	}
	return time.Parse(time.RFC3339, *s)
}

func ParseDateTime(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, nil
	}
	return time.ParseInLocation(DateTime, s, time.Local)
}

func ParseDateTimeUTC(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, nil
	}
	t, err := ParseDateTime(s)
	if err != nil {
		return time.Time{}, err
	}
	return t.UTC(), nil
}

func ParseModelDateTime(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, nil
	}
	res, err := time.ParseInLocation(time.RFC3339, s, time.Local)
	if err != nil {
		res, err = time.ParseInLocation(DateTime, s, time.Local)
	}

	return res, err
}

func FormatTimeStr(s string) string {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return s
	}
	layout := "2006-01-02 15:04:05 -07"
	return t.Format(layout)
}

func FormatTime(t time.Time) string {
	return t.Format(DateTime)
}

func FormatDate(t time.Time) string {
	return t.Format(Date)
}

func CostMs(begin time.Time) int64 {
	return time.Since(begin).Milliseconds()
}

// Cost is the readable time duration in seconds, e.g. 5m15s.
func Cost(begin time.Time) string {
	return time.Since(begin).Truncate(time.Second).String()
}
