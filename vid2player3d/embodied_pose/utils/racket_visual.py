"""Small XML-based racket outlines for the display-free test recorder."""
import xml.etree.ElementTree as ET

import numpy as np


def racket_lines(asset_file):
    """Read handle and string-bed dimensions in the physical Racket body frame."""
    body = ET.parse(asset_file).find(".//body[@name='Racket']")
    if body is None:
        return []
    lines = []
    for geom in body.findall('geom'):
        if geom.get('type') != 'cylinder' or 'fromto' not in geom.attrib:
            raise ValueError('Racket debug rendering requires cylinder fromto geometry')
        ends = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
        radius = float(geom.get('size').split()[0])
        axis = ends[1] - ends[0]
        length = np.linalg.norm(axis)
        axis /= length
        if length > 2 * radius:
            lines.append(ends)  # Handle centerline.
        else:
            basis = np.eye(3)[np.argmin(np.abs(axis))]
            u = np.cross(axis, basis)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            theta = np.linspace(0, 2 * np.pi, 33)
            lines.append(ends.mean(0) + radius * (np.cos(theta)[:, None] * u
                                                  + np.sin(theta)[:, None] * v))
    return lines


def transform_lines(lines, position, rotation_xyzw):
    """Transform outlines with XYZW quaternions, including q/-q equivalence."""
    q = np.asarray(rotation_xyzw)
    q = q / np.linalg.norm(q)
    result = []
    for points in lines:
        uv = np.cross(q[:3], points)
        result.append(position + points + 2 * (q[3] * uv + np.cross(q[:3], uv)))
    return result


def wrist_racket_lines(lines, wrist_position, wrist_rotation_xyzw, offset):
    """XML-local outlines follow the actual or reference wrist, in the same frame.

    IsaacGym may rebase a fixed body's frame during MJCF import. Its raw racket
    quaternion therefore cannot directly transform geometry from the XML frame.
    Supported fixed assets have identity body rest rotations, so constructing
    the attachment from the wrist preserves the XML geometry exactly.
    """
    return transform_lines([line + offset for line in lines],
                           wrist_position, wrist_rotation_xyzw)


def racket_face_normal(asset_file):
    """String-bed cylinder axis in the validated identity-rest XML wrist frame."""
    body = ET.parse(asset_file).find(".//body[@name='Racket']")
    if body is not None:
        for geom in body.findall('geom'):
            if geom.get('type') == 'cylinder' and 'fromto' in geom.attrib:
                ends = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
                axis = ends[1] - ends[0]
                length = np.linalg.norm(axis)
                if 0 < length <= 2 * float(geom.get('size').split()[0]):
                    return axis / length
    raise ValueError('Expected a string-bed cylinder for racket tracking')
