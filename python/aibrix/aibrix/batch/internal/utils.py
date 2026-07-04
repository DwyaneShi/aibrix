# Copyright 2026 The Aibrix Team.
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

import asyncio
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, Union

Exceptions = Union[Type[Exception], Tuple[Type[Exception], ...]]


def async_retry(
    exceptions: Exceptions = Exception,
    *,
    tries: int = 3,
    delay: float = 0,
    backoff: float = 1,
    logger: Optional[Any] = None,
) -> Callable:
    """async retry decorator"""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _tries, _delay = tries, delay
            while _tries:
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    _tries -= 1
                    if not _tries:
                        raise
                    if logger is not None:
                        logger.warning("%s, retrying in %s seconds...", e, _delay)
                    await asyncio.sleep(_delay)
                    _delay *= backoff

        return wrapper

    return decorator
