"""Shared court geometry conventions, independent of simulator and rendering.

The existing trajectory models use a uniform net height of 1.07 m. This is
conservative at court center, rather than a sagging-net model or collision asset.
Ball size belongs to the active BallPhysics settings, not court geometry.
"""

NET_HEIGHT_M = 1.07
