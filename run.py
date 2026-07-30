import asyncio
from cli import run_cli_session

if __name__ == "__main__":
    try:
        asyncio.run(run_cli_session())
    except KeyboardInterrupt:
        print("\nShutdown.")
