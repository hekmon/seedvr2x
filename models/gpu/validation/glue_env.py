"""The paths of this folder's Python scripts, read from the environment that glue.env sets
(glue.env.example names every one): env(NAME) returns one, or stops the script, naming it."""

import os
import sys


def env(name):
    """The value of the environment variable NAME; exits with status 2 when it is unset or empty."""
    value = os.environ.get(name, "")
    if not value:
        sys.stderr.write(
            f"{os.path.basename(sys.argv[0])}: {name} is unset: source your glue.env "
            "(models/gpu/validation/glue.env.example)\n"
        )
        sys.exit(2)
    return value
