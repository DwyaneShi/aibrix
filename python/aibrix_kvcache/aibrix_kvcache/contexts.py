# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextvars
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


def generate_context_manager(context_var: contextvars.ContextVar[T]):
    """
    Generates a context manager for the given ContextVar.

    Args:
        context_var: The ContextVar instance to create a context manager for

    Returns:
        A context manager that sets the contextvar's value within its scope
    """

    @contextmanager
    def var_context(value: T, enabled: bool = True):
        if enabled:
            token = context_var.set(value)
            try:
                yield
            finally:
                context_var.reset(token)
        else:
            yield

    return var_context


# context variables
# layer_id: the id of the current layer. If it is -1, it means that the current
# operation is not a per-layer operation, i.e., the current operation will use
# all the layers configured in the cache manager.
layer_id: contextvars.ContextVar[int] = contextvars.ContextVar(
    "layer_id", default=-1
)

# contexts corresponding to context variables
layer_context = generate_context_manager(layer_id)
