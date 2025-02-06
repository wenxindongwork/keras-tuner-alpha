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

import argparse
import pkg_resources
import sys

def main():
    parser = argparse.ArgumentParser(description="Kithara CLI tool")
    subparsers = parser.add_subparsers(dest="command")

    # Dynamically load all commands from entry points
    for entry_point in pkg_resources.iter_entry_points("kithara.commands"):
        command_parser = subparsers.add_parser(entry_point.name)
        command_func = entry_point.load()
        # Since the entry point directly loads the main function
        if hasattr(command_func, 'setup_parser'):
            command_func.setup_parser(command_parser)
        command_parser.set_defaults(func=command_func)

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)  # Pass the parsed args to the command function

if __name__ == "__main__":
    main()
