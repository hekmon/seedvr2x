"""What a brick of the library refuses to run (DESIGN.md, Repository and apps): the library's own
base for its refusals, so that it imports nothing of the command line's job. Plain Python, no
torch."""


class Refusal(ValueError):
    """What a brick refuses to run: the message says what, for the user. The command line reports
    it as it reports a job it refuses.

    A refusal that an application other than the command line must word for itself (the image
    server answers a refused picture with the reason and a way out of its own, DESIGN.md, The
    image server, Errors) carries its message in parts as well, all keyword-only and "" where a
    refusal has none; str() is the message, whole, whatever the parts:

    - kind: a stable lower-case slug naming the refusal ("raw", "too-large"); an application
      words its own way out by it. The kinds of a brick are listed in the brick (picture.KINDS).
    - reason: what is refused and why, a complete clause naming no file, no command and no option
      of the command line ("the picture has transparent pixels, and seedvr2x refuses it: ...").
    - way: the command line's way out, as its message has it, which may name the file and give an
      ffmpeg command; "" where the message has none."""

    def __init__(self, message: str, *, kind: str = "", reason: str = "", way: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.reason = reason
        self.way = way


class JobError(Refusal):
    """A job seedvr2x can't lay out: a bad cut list; the message says what."""
