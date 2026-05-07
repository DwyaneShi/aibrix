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
package json

import (
	"bytes"
	"encoding/json"
)

type ParserImpl struct{}

func NewParserImpl() *ParserImpl {
	return &ParserImpl{}
}

func (p ParserImpl) Marshal(v any) ([]byte, error) {
	return json.Marshal(v)
}

func (p ParserImpl) MarshalToString(v any) (string, error) {
	bytes, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func (p ParserImpl) Unmarshal(data []byte, v any) error {
	return json.Unmarshal(data, v)
}

func (p ParserImpl) UnmarshalUseNumber(data []byte, v any) error {
	decoder := json.NewDecoder(bytes.NewBuffer(data))
	decoder.UseNumber()
	return decoder.Decode(v)
}

func (p ParserImpl) UnmarshalFromString(str string, v any) error {
	return json.Unmarshal([]byte(str), v)
}
