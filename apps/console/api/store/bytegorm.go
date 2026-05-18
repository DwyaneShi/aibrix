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

package store

import (
	"fmt"
	"net/url"
	"strings"
	"time"

	"code.byted.org/gorm/bytedgorm"
	"github.com/go-sql-driver/mysql"
	"github.com/vllm-project/aibrix/apps/console/api/error_injection"
	"gorm.io/gorm"
)

const (
	maxRetryTime   = 3
	maxIdleConns   = 10
	maxOpenConns   = 300
	defaultTimeout = 10 * time.Second
)

func NewByteRDSStore(dsn, encryptionKey string, injector error_injection.Injector) (*GORMStore, error) {
	cfg, err := mysql.ParseDSN(dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to parse dsn: %s, error: %w", dsn, err)
	}

	// parse addr, which is in the format of "psm:port"
	addr := strings.Split(cfg.Addr, ":")
	psm := addr[0]
	dsnParams := url.Values{}
	for k, v := range cfg.Params {
		dsnParams.Set(k, v)
	}

	if cfg.ReadTimeout == 0 {
		cfg.ReadTimeout = defaultTimeout
	}

	if cfg.WriteTimeout == 0 {
		cfg.WriteTimeout = defaultTimeout
	}

	if cfg.Timeout == 0 {
		cfg.Timeout = defaultTimeout
	}

	var db *gorm.DB
	for range maxRetryTime {
		db, err = gorm.Open(
			bytedgorm.MySQL(psm, cfg.DBName).With(func(conf *bytedgorm.DBConfig) {
				conf.DSNParams = dsnParams
				conf.ReadTimeout = cfg.ReadTimeout
				conf.WriteTimeout = cfg.WriteTimeout
				conf.Timeout = cfg.Timeout
			}).WithReadReplicas(),
			bytedgorm.WithDefaults(),
			bytedgorm.ConnPool{MaxIdleConns: maxIdleConns, MaxOpenConns: maxOpenConns},
		)

		if err != nil {
			time.Sleep(time.Second)
			continue
		}

		if db == nil {
			err = fmt.Errorf("service discovery failed")
			break
		}

		break
	}

	if err != nil {
		return nil, fmt.Errorf("failed to open rds, dsn: %s, error: %w", dsn, err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("failed to access rds, dsn: %s, error: %w", dsn, err)
	}

	if err = sqlDB.Ping(); err != nil {
		_ = sqlDB.Close()
		return nil, fmt.Errorf("failed to ping rds, dsn: %s, error: %w", dsn, err)
	}

	s, err := newGORMStore(db, encryptionKey, injector)
	if err != nil {
		_ = sqlDB.Close()
		return nil, err
	}

	// ByteRDS does not support migrations, so we skip it
	return s, nil
}
