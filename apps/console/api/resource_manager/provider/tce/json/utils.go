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

func Marshal(v any) ([]byte, error) {
	return DefaultParser.Marshal(v)
}

func MarshalToString(v any) (string, error) {
	return DefaultParser.MarshalToString(v)
}

func Unmarshal(bytes []byte, v any) error {
	return DefaultParser.Unmarshal(bytes, v)
}

func UnmarshalUseNumber(bytes []byte, v any) error {
	return DefaultParser.UnmarshalUseNumber(bytes, v)
}

func UnmarshalFromString(str string, v any) error {
	return DefaultParser.UnmarshalFromString(str, v)
}
