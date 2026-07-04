import sys

from cyberwheel.profiler.cli import EXIT_ERROR, main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"profiler error: {error!r}", file=sys.stderr)
        sys.exit(EXIT_ERROR)
