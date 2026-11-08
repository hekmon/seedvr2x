"""seedvr2x's lines in ComfyUI's console: "[seedvr2x] " before each line that seedvr2x's loggers
send there, the library's (`seedvr2x.<module>`) and the nodes' (`seedvr2x_comfyui.<module>`), so
that a user reading a console shared with every other node sees which lines are seedvr2x's (the
user, 2026-11-08). In this process alone: the command line and the image server log through their
own handlers, with the time, the level and the logger's name.

ComfyUI's console is the root logger's handlers (v0.44.0, app/logger.py's setup_logger: a stream
handler and a formatter of the message alone, set before any custom node is imported), which
every other node's lines go through too, so nothing of theirs is changed: a root handler's filter
would be ComfyUI's to set, and a logger's filters run for the records made on that logger only,
not for its children's. A record goes to the handlers of its logger, then of each parent, the
same record, up to the root (logging's Logger.callHandlers), so the two loggers at the top of
seedvr2x's names each have one handler (Prefixing) that writes nothing and puts the prefix on the
record itself as it passes, before the root's handlers take it: their levels, filters and formats
apply as before, and the loggers propagate as before. Read in ComfyUI's and CPython's sources, not
copied (GPL-3.0).

One record a line (the user, 2026-11-08): ComfyUI's formatter puts the level ("[INFO] ") before a
record's first line only, so a text of several lines (the plan, the cuts, a refusal and its way
out) is said a line a record (`say`), each with its level and the prefix.

The library's checks of the model files at DEBUG (the user, 2026-11-08: the cache's line, each
file's path in it and its two pin checks, said three and two times in one run): the Models node
says one line a file, where it is and how large, and the library's INFO records of QUIET are
DEBUG on their way to ComfyUI's console (its console shows INFO and up; `--verbose DEBUG` shows
them, and so does a file log at DEBUG, `--verbose DEBUG FILE`: v0.44.0 keeps no file log by
default). The command line and the server keep them at INFO."""

import logging

PREFIX = "[seedvr2x] "
LOGGERS = ("seedvr2x", "seedvr2x_comfyui")

# The library's loggers whose INFO records are DEBUG in ComfyUI's console: the model files found,
# pulled and checked against their pins (pull.py), their headers checked (weights.py).
QUIET = ("seedvr2x.pull", "seedvr2x.weights")


def prefixed(text: str) -> str:
    """`text` with PREFIX before each of its lines (a record that holds several still reads as
    seedvr2x's in every one; the nodes' own say a line a record, `say`)."""
    return "\n".join(PREFIX + line for line in text.split("\n"))


def say(logger: logging.Logger, level: int, text: str) -> None:
    """`text` on `logger` at `level`, one record a line, in order, so that each line in ComfyUI's
    console carries its level and the prefix. Each record holds the whole text too
    (`seedvr2x_text`), for a handler that reads it whole (the tests')."""
    for line in text.split("\n"):
        logger.log(level, "%s", line, extra={"seedvr2x_text": text}, stacklevel=2)


class Prefixing(logging.Handler):
    """The handler of seedvr2x's two loggers in ComfyUI's process: each record's message made and
    prefixed (prefixed) in the record, once, which then goes on to the root's handlers; an INFO
    record of QUIET's made DEBUG first, so that the root's handlers at INFO leave it out."""

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "seedvr2x_prefixed", False):
            return
        try:
            if record.name in QUIET and record.levelno == logging.INFO:
                record.levelno, record.levelname = (
                    logging.DEBUG,
                    logging.getLevelName(logging.DEBUG),
                )
            record.msg = prefixed(record.getMessage())
            record.args = None
            record.seedvr2x_prefixed = True
        except Exception:
            self.handleError(record)


def install() -> None:
    """The prefix on from now on, once however often it is asked (ComfyUI imports the folder
    once; the tests import it again): one Prefixing handler on each of the two loggers."""
    for name in LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(each, Prefixing) for each in logger.handlers):
            logger.addHandler(Prefixing())
