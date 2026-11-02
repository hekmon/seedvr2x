"""What a brick of the library refuses to run (DESIGN.md, Repository and apps): the library's own
base for its refusals, so that it imports nothing of the command line's job. Plain Python, no
torch."""


class Refusal(ValueError):
    """What a brick refuses to run: the message says what, for the user. The command line reports
    it as it reports a job it refuses."""
