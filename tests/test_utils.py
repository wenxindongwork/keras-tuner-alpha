"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import signal
from functools import wraps

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")

            # Set the timeout handler
            original_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the original handler and disable the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
            return result
        return wrapper
    return decorator
