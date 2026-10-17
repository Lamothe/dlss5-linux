#!/usr/bin/env python3
"""Keep newly created X11 windows on the primary display.

Test runs (Proton/Wine probes, the DLSS sample app, benchmark runs) create
windows that the desktop may place on the secondary SONY screen. This guard
enumerates the root window's children before a run starts, then polls during
the run and moves any *newly created* top-level window onto the primary
display (the Dell, 3440x1440 at 0,0). Pre-existing windows — e.g. a video
player on the secondary screen — are never touched.

Usage: x11-keep-on-primary.py [SECONDS]   (default 120)

Run it in the background around a test invocation and stop it afterwards:

    python3 vr/x11-keep-on-primary.py 90 & guard=$!
    bash vr/run-feature-probe.sh ladder sNN NN
    kill "$guard"

Pure ctypes/Xlib; no third-party packages. Fail-closed: any X11 error stops
the guard rather than touching unknown windows.
"""

import ctypes
import ctypes.util
import math
import os
import sys
import time

# The primary display (Dell 34" 3440x1440 at 0,0); the secondary SONY starts
# at x=3440. New windows land at this safe point inside the primary display.
PRIMARY_X = 100
PRIMARY_Y = 100

POLL_SECONDS = 0.25


class Guard:
    def __init__(self) -> None:
        name = ctypes.util.find_library("X11")
        if name is None:
            raise RuntimeError("libX11 not found")
        self.lib = ctypes.CDLL(name)
        self.lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.lib.XOpenDisplay.restype = ctypes.c_void_p
        self.lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        self.lib.XDefaultRootWindow.restype = ctypes.c_ulong
        self.lib.XQueryTree.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint)]
        self.lib.XQueryTree.restype = ctypes.c_int
        self.lib.XMoveWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
        self.lib.XMoveWindow.restype = ctypes.c_int
        self.lib.XFlush.argtypes = [ctypes.c_void_p]
        self.lib.XFlush.restype = ctypes.c_int
        self.lib.XFree.argtypes = [ctypes.c_void_p]
        self.lib.XFree.restype = ctypes.c_int
        self.lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.lib.XCloseDisplay.restype = ctypes.c_int

        display = self.lib.XOpenDisplay(os.environb.get(b"DISPLAY"))
        if not display:
            raise RuntimeError("cannot open X display")
        self.display = display
        self.root = self.lib.XDefaultRootWindow(display)

    def children(self) -> set[int]:
        root = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        if not self.lib.XQueryTree(
                self.display, self.root, ctypes.byref(root),
                ctypes.byref(parent), ctypes.byref(children),
                ctypes.byref(count)):
            raise RuntimeError("XQueryTree failed")
        ids = {children[i] for i in range(count.value)}
        if children:
            self.lib.XFree(children)
        return ids

    def move(self, window: int) -> None:
        self.lib.XMoveWindow(self.display, window, PRIMARY_X, PRIMARY_Y)
        self.lib.XFlush(self.display)

    def close(self) -> None:
        self.lib.XCloseDisplay(self.display)


def main() -> int:
    try:
        seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    except ValueError:
        seconds = 0.0
    if not math.isfinite(seconds) or seconds <= 0:
        print("usage: x11-keep-on-primary.py [SECONDS]", file=sys.stderr)
        return 2
    try:
        guard = Guard()
    except RuntimeError as error:
        print(f"guard: {error}", file=sys.stderr)
        return 1
    try:
        seen = guard.children()
        print(f"guard: watching {len(seen)} existing windows for "
              f"{seconds:.0f}s", flush=True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            for window in sorted(guard.children() - seen):
                seen.add(window)
                guard.move(window)
                print(f"guard: moved new window 0x{window:x} to "
                      f"({PRIMARY_X},{PRIMARY_Y})", flush=True)
    except RuntimeError as error:
        print(f"guard: {error}", file=sys.stderr)
        return 1
    finally:
        guard.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
