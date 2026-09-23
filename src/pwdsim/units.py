"""Unit conversion factors.

pwdsim works in SI units internally.  Multiply by these constants to convert a
quantity into SI, and divide to convert back out, e.g. ``7 * INCH`` is 7 inches in
meters and ``length / FOOT`` is ``length`` in feet.
"""

import math

# Length (to meters)
METER = 1.0
CENTIMETER = 1.0e-2
MILLIMETER = 1.0e-3
INCH = 0.0254
FOOT = 12 * INCH

# Mass (to kilograms)
KILOGRAM = 1.0
GRAM = 1.0e-3
OUNCE = 0.028349523125

# Angle (to radians)
RADIAN = 1.0
DEGREE = math.pi / 180

LENGTH_UNITS = {
    "m": METER,
    "cm": CENTIMETER,
    "mm": MILLIMETER,
    "in": INCH,
    "ft": FOOT,
}
"""Length units by abbreviation, as used by the plotting routines."""
