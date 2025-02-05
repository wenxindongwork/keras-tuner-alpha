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
