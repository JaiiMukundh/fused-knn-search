#!/usr/bin/env python3
"""Wrap remote/kaggle_bootstrap.py into remote/kaggle_bootstrap.ipynb.

Kaggle's "File -> Import Notebook" accepts a plain .ipynb. Keeping the real
source as a .py (and generating the notebook) means the bootstrap stays easy
to diff, review and edit -- no JSON juggling in a browser.

Usage
-----
    python3 remote/make_ipynb.py

Then upload remote/kaggle_bootstrap.ipynb to Kaggle (or open the .py content
and paste it into one cell -- both work; the notebook path is less error
prone on copy/paste).
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "kaggle_bootstrap.py"
DST = HERE / "kaggle_bootstrap.ipynb"


def main() -> int:
    text = SRC.read_text()
    # nbformat stores "source" as a list of lines, each keeping its own "\n".
    source = text.splitlines(keepends=True)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# PCAP Kaggle bootstrap\n",
                    "\n",
                    "Run the single cell below once per Kaggle session\n",
                    "(Settings: **Accelerator = GPU T4 x2**, **Internet = on**).\n",
                    "\n",
                    "It prints your ssh command, an `~/.ssh/config` entry and the\n",
                    "`remote/kaggle.env` block for the local `remote/kremote.sh`.\n",
                    "\n",
                    "Full walkthrough + troubleshooting: **remote/TUNNEL_SETUP.md**.\n",
                    "\n",
                    "Edit `remote/kaggle_bootstrap.py`, not this notebook:\n",
                    "`python3 remote/make_ipynb.py` regenerates it.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    DST.write_text(json.dumps(nb, indent=1) + "\n")

    # The generated notebook must round-trip to the exact source file.
    back = "".join(json.loads(DST.read_text())["cells"][1]["source"])
    assert back == text, "notebook source does not round-trip to kaggle_bootstrap.py"
    print(f"wrote {DST} ({len(source)} source lines, round-trip OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
