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

package store

import (
	"testing"

	"github.com/go-sql-driver/mysql"
)

func TestByteRDSDSNParamsRepinUTCFieldsRemovedByParseDSN(t *testing.T) {
	cfg, err := mysql.ParseDSN("@tcp(test-psm:3306)/aibrix?cluster=lf&loc=UTC&parseTime=true")
	if err != nil {
		t.Fatalf("parse DSN: %v", err)
	}

	params := byteRDSDSNParams(cfg)
	for key, want := range map[string]string{
		"cluster":   "lf",
		"loc":       "UTC",
		"parseTime": "true",
	} {
		if got := params.Get(key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}
}
