#!/usr/bin/env python3

import importlib.resources
from . import data, plotting, model
STYLE_FILE = importlib.resources.files('workshop_utils') / 'nemos.mplstyle'
