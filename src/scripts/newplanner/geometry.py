

class Polygon:
    def __init__(self, points):
        self.points = points

    def intersects(self, other):
        edges1 = self._get_edges()
        edges2 = other._get_edges()
        for e1 in edges1:
            for e2 in edges2:
                if self._do_edges_intersect(e1, e2):
                    return True
        return False

    def _get_edges(self):
        edges = []
        for i in range(len(self.points)):
            edges.append((self.points[i], self.points[(i+1) % len(self.points)]))
        return edges

    def _do_edges_intersect(self, e1, e2):
        p1, q1 = e1
        p2, q2 = e2
        p1x, p1y, p1z = p1
        q1x, q1y, q1z = q1
        p2x, p2y, p2z = p2
        q2x, q2y, q2z = q2
        dx1 = q1x - p1x
        dy1 = q1y - p1y
        dz1 = q1z - p1z
        dx2 = q2x - p2x
        dy2 = q2y - p2y
        dz2 = q2z - p2z
        delta = dx1 * dy2 - dy1 * dx2
        if delta == 0:
            return False
        s = (dx2 * (p1y - p2y) - dy2 * (p1x - p2x)) / delta
        t = (dx1 * (p1y - p2y) - dy1 * (p1x - p2x)) / delta
        return (0 <= s <= 1) and (0 <= t <= 1) and (p1z + s*dz1 == p2z + t*dz2)
