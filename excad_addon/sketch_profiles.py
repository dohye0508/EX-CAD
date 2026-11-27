# SPDX-License-Identifier: GPL-2.0-or-later
"""Sample procedural sketch entities and join their shared or coincident endpoints."""

import math

from mathutils import Vector

TAU = 2 * math.pi


def cross(a, b):
    return a.x * b.y - a.y * b.x


def signed_area(loop):
    return sum(cross(a, loop[(i + 1) % len(loop)]) for i, a in enumerate(loop)) / 2


def circle_points(center, radius, start, span, steps, endpoint):
    return [center + radius * Vector((math.cos(start + span * i / steps),
                                      math.sin(start + span * i / steps)))
            for i in range(steps + int(endpoint))]


def circumcenter(a, b, c):
    u, v = b - a, c - a
    det = 2 * cross(u, v)
    if abs(det) <= 1e-10 * max(u.length_squared, v.length_squared, 1e-12):
        raise ValueError("A three-point arc needs noncollinear points")
    return a + Vector(((v.y * u.length_squared - u.y * v.length_squared) / det,
                       (u.x * v.length_squared - v.x * u.length_squared) / det))


def offset_loop(loop, distance):
    """Intersect adjacent parallel edges; positive distance always grows the profile."""
    winding = 1 if signed_area(loop) > 0 else -1
    result = []
    for i, p in enumerate(loop):
        a = (p - loop[i - 1]).normalized()
        b = (loop[(i + 1) % len(loop)] - p).normalized()
        na, nb = Vector((a.y, -a.x)) * winding, Vector((b.y, -b.x)) * winding
        denom = 1 + na.dot(nb)
        if denom < 1e-7:
            raise ValueError("Offset has a reversed or degenerate corner")
        result.append(p + (na + nb) * (distance / denom))
    if any((result[(i + 1) % len(loop)] - result[i]).dot(loop[(i + 1) % len(loop)] - p) <= 0
           for i, p in enumerate(loop)):
        raise ValueError("Offset collapses the profile")
    return result


def _spline(points, closed, steps):
    points = ([points[-1]] + points + points[:2] if closed
              else [points[0]] + points + [points[-1]])
    result = []
    for i in range(1, len(points) - 2):
        a, b, c, d = points[i - 1:i + 3]
        for j in range(steps):
            t = j / steps
            result.append(0.5 * (2 * b + (-a + c) * t
                                 + (2*a - 5*b + 4*c - d) * t*t
                                 + (-a + 3*b - 3*c + d) * t*t*t))
    if not closed:
        result.append(points[-2])
    return result


def sample(doc, entity, segments=64):
    point = lambda index: Vector((doc.points[index].u, doc.points[index].v))
    kind, args = entity.etype, entity.params
    a = point(entity.p0)
    if kind == "LINE":
        return [a, point(entity.p1)], False
    if kind in {"CIRCLE", "REGULAR_POLYGON"}:
        count = args.get("sides", segments)
        return circle_points(a, entity.radius, math.radians(args.get("rotation", 0)), TAU, count, False), True
    if kind == "ARC":
        if "through" in args:
            middle, end = point(args["through"]), point(entity.p1)
            center = circumcenter(a, middle, end)
            start = math.atan2(a.y - center.y, a.x - center.x)
            span = (math.atan2(end.y - center.y, end.x - center.x) - start) % TAU
            if (math.atan2(middle.y - center.y, middle.x - center.x) - start) % TAU > span:
                span -= TAU
            radius = (a - center).length
        else:
            center, radius = a, entity.radius
            start, span = math.radians(args["start_deg"]), math.radians(args["angle_deg"])
        steps = max(2, math.ceil(abs(span) / TAU * segments))
        return circle_points(center, radius, start, span, steps, True), False
    if kind == "SLOT":
        b = point(entity.p1)
        angle = math.atan2(b.y - a.y, b.x - a.x)
        half = max(4, segments // 2)
        return (circle_points(b, entity.radius, angle - math.pi/2, math.pi, half, True)
                + circle_points(a, entity.radius, angle + math.pi/2, math.pi, half, True)), True
    if kind == "SPLINE":
        return _spline([point(i) for i in args["points"]], args["closed"], args["steps"]), args["closed"]
    raise ValueError(f"Unsupported sketch entity {kind}")


def join_paths(paths):
    """Join independent sketch lines/arcs as well as entities sharing point IDs."""
    remaining = [list(p) for p in paths if len(p) > 1]
    loops = []
    while remaining:
        chain = remaining.pop(0)
        while (chain[-1] - chain[0]).length > 1e-6:
            for i, path in enumerate(remaining):
                if (path[0] - chain[-1]).length <= 1e-6:
                    chain.extend(path[1:])
                elif (path[-1] - chain[-1]).length <= 1e-6:
                    chain.extend(reversed(path[:-1]))
                else:
                    continue
                remaining.pop(i)
                break
            else:
                break
        if len(chain) >= 4 and (chain[0] - chain[-1]).length <= 1e-6:
            loops.append(chain[:-1])
    return loops


def paths_and_loops(doc, segments=64):
    closed, paths, drawing = [], [], []
    for entity in doc.entities:
        args = entity.params
        if entity.etype in {"OFFSET_LOOP", "MIRROR2D"}:
            profiles = closed + join_paths(paths)
            index = args["source"]
            if not 0 <= index < len(profiles):
                raise ValueError("Referenced profile is missing or open")
            loop = profiles[index]
            if entity.etype == "OFFSET_LOOP":
                points = offset_loop(loop, args["distance"])
            else:
                axis = args["axis"]
                points = [Vector((v.x, 2*args["offset"] - v.y)) if axis == "X"
                          else Vector((2*args["offset"] - v.x, v.y)) for v in reversed(loop)]
            is_closed = True
        else:
            points, is_closed = sample(doc, entity, segments)
        drawing.append((entity, points, is_closed))
        (closed if is_closed else paths).append(points)
    return drawing, closed + join_paths(paths)
