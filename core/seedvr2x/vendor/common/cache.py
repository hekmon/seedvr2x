# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# Modified for seedvr2x: the shapes' small values held for a whole forward, made after a poll.

import threading
from typing import Callable

# seedvr2x: `hooks.missed`, the calling thread's, is called before a held value is first made
# (Cache.held; model.py's _paced sets it to its poll of the device for the forward it runs),
# absent for nothing: what makes a value copies shapes and lengths from the device, which waits
# in C for the work queued before it, out of a stop's reach (DESIGN.md, Allocator, "Out of the
# stop's reach"); after the poll it waits for its own few index kernels alone. Per thread, so
# that two forwards on two threads (two DiTs) poll their own cards.
hooks = threading.local()


class Cache:
    """Caching reusable args for faster inference"""

    def __init__(self, disable=False, prefix="", cache=None):
        self.cache = cache if cache is not None else {}
        self.disable = disable
        self.prefix = prefix

    def __call__(self, key: str, fn: Callable):
        if self.disable:
            return fn()

        key = self.prefix + key
        try:
            result = self.cache[key]
        except KeyError:
            result = fn()
            self.cache[key] = result
        return result

    def held(self, key: str, fn: Callable):
        """seedvr2x: `fn()` made once for the forward this cache is, under `key`, whatever
        `disable`: for the shapes' small values (window indices, lengths, the sizes a kernel
        needs on the host), which every block made again with the cache disabled, as the
        pipeline runs it, each making a wait on the device in C. Never for a value of a
        token's size (the modulation's repeated embedding, RoPE's frequencies): held to the
        forward's end, it would add to the DiT's peak."""
        key = self.prefix + key
        try:
            return self.cache[key]
        except KeyError:
            missed = getattr(hooks, "missed", None)
            if missed is not None:
                missed()
            result = fn()
            self.cache[key] = result
            return result

    def namespace(self, namespace: str):
        return Cache(
            disable=self.disable,
            prefix=self.prefix + namespace + ".",
            cache=self.cache,
        )

    def get(self, key: str):
        key = self.prefix + key
        return self.cache[key]
