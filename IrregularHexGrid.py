import numpy as np
import matplotlib.pyplot as plt
import math
import random

# --------------------------
# Classes de base
# --------------------------

class Vertex:
    _vertices = {}
    
    def __init__(self, point):
        self.point = tuple(point)
        self.is_outer = False
        self.faces = []

    @staticmethod
    def key(point, precision=8):
        return (round(point[0], precision), round(point[1], precision))
    
    @staticmethod
    def get(point):
        key = Vertex.key(point)
        if key in Vertex._vertices:
            return Vertex._vertices[key]
        else:
            vertex = Vertex(point)
            Vertex._vertices[key] = vertex
            return vertex

class Edge:
    _edges = {}

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.faces = []

    @staticmethod
    def get(a, b):
        if a.point > b.point:
            a, b = b, a
        key = (id(a), id(b))
        if key in Edge._edges:
            return Edge._edges[key]
        else:
            edge = Edge(a, b)
            Edge._edges[key] = edge
            return edge
        
    def add_face(self, face):
        self.faces.append(face)

    def get_center(self):
        return ((self.a.point[0] + self.b.point[0])/2, (self.a.point[1] + self.b.point[1])/2)

class Face:
    _faces = {}

    def __init__(self, vertices):
        self.vertices = vertices
        self.edges = []
    
    @staticmethod
    def triangle(a, b, c):
        f = Face([a, b, c])
        f.edges = [Edge.get(a, b), Edge.get(b, c), Edge.get(c, a)]
        for e in f.edges:
            e.add_face(f)
        return f

    @staticmethod
    def quad(a, b, c, d):
        f = Face([a, b, c, d])
        f.edges = [Edge.get(a, b), Edge.get(b, c), Edge.get(c, d), Edge.get(d, a)]
        for e in f.edges:
            e.add_face(f)
        return f

    def get_centroid(self):
        x = sum(v.point[0] for v in self.vertices)/len(self.vertices)
        y = sum(v.point[1] for v in self.vertices)/len(self.vertices)
        return [x, y]

    def subdivide(self):
        o = Vertex.get(self.get_centroid())
        if len(self.vertices) == 3:
            a, b, c = self.vertices
            x = Vertex.get(self.edges[0].get_center())
            y = Vertex.get(self.edges[1].get_center())
            z = Vertex.get(self.edges[2].get_center())
            return [
                Face.quad(a, x, o, z),
                Face.quad(b, y, o, x),
                Face.quad(c, z, o, y)
            ]
        elif len(self.vertices) == 4:
            a, b, c, d = self.vertices
            x = Vertex.get(self.edges[0].get_center())
            y = Vertex.get(self.edges[1].get_center())
            z = Vertex.get(self.edges[2].get_center())
            w = Vertex.get(self.edges[3].get_center())
            return [
                Face.quad(a, x, o, w),
                Face.quad(b, y, o, x),
                Face.quad(c, z, o, y),
                Face.quad(d, w, o, z)
            ]

# --------------------------
# Fonctions utilitaires
# --------------------------

def lerp2d(p0, p1, t):
    return [(1-t)*p0[0]+t*p1[0], (1-t)*p0[1]+t*p1[1]]

def HexPoint(radius, point_i):
    angle = -math.pi/2 + math.pi/3 * point_i
    return [radius * math.cos(angle), radius * math.sin(angle)]

def get_arithmetic_progression_sum(a, d, n):
    return int(n*(2*a + (n-1)*d)/2)

# --------------------------
# Hex Grid Generation
# --------------------------

def generate_hex_grid_vertices(ring_count, ring_radius):
    point_count = get_arithmetic_progression_sum(1,6,ring_count+1) - ring_count
    points = [None]*point_count
    points[0] = [0,0]
    point_index = 1
    for ring_i in range(1, ring_count+1):
        r = ring_radius*ring_i
        for point_i in range(6):
            start = HexPoint(r, point_i)
            end = HexPoint(r, point_i+1)
            for side_point_i in range(ring_i):
                t = side_point_i/ring_i
                points[point_index] = lerp2d(start, end, t)
                point_index +=1
    return points

def get_vertex_index_by_ring(ring_index, point_index):
    return 1 + get_arithmetic_progression_sum(0,6,ring_index) + point_index

def triangulate_hex_grid_vertices(ring_count, vertices):
    triangle_count = 6*get_arithmetic_progression_sum(1,2,ring_count)
    triangles = [None]*triangle_count
    tri_i = 0
    for ring_i in range(1, ring_count+1):
        point_count = 6*ring_i
        prev_point_count = 6*(ring_i-1)
        side_count = ring_i
        for point_i in range(point_count):
            a = get_vertex_index_by_ring(ring_i, point_i)
            b = get_vertex_index_by_ring(ring_i, (point_i+1)%point_count)
            c = 0
            if ring_i>1:
                c = get_vertex_index_by_ring(ring_i-1, (point_i - point_i//side_count)%prev_point_count)
            triangles[tri_i] = [a,b,c]
            tri_i+=1
            if ring_i<ring_count:
                c2 = get_vertex_index_by_ring(ring_i+1, point_i+1 + point_i//side_count)
                triangles[tri_i] = [a,c2,b]
                tri_i+=1
    return triangles

# --------------------------
# Merge + subdivision aléatoire
# --------------------------

def merge_and_subdivide(triangles, points):
    unused = [Face.triangle(Vertex.get(points[i[0]]),
                            Vertex.get(points[i[1]]),
                            Vertex.get(points[i[2]])) for i in triangles]

    edge_to_faces = {}
    for idx, f in enumerate(unused):
        for e in f.edges:
            key = tuple(sorted([id(e.a), id(e.b)]))
            edge_to_faces.setdefault(key, []).append(idx)

    faces = []
    used = set()

    indices = list(range(len(unused)))
    random.shuffle(indices)

    for i in indices:
        if i in used: 
            continue
        f = unused[i]
        neighbor_idx = None
        for e in f.edges:
            key = tuple(sorted([id(e.a), id(e.b)]))
            for adj in edge_to_faces[key]:
                if adj != i and adj not in used:
                    shared = list(set(f.vertices).intersection(set(unused[adj].vertices)))
                    if len(shared) == 2:
                        neighbor_idx = adj
                        break
            if neighbor_idx is not None:
                break

        if neighbor_idx is not None:
            f2 = unused[neighbor_idx]
            v1, v2 = set(f.vertices), set(f2.vertices)
            shared = list(v1.intersection(v2))
            unique1 = list(v1 - set(shared))[0]
            unique2 = list(v2 - set(shared))[0]
            quad_face = Face.quad(unique1, shared[0], unique2, shared[1])
            faces.extend(quad_face.subdivide())
            used.add(i)
            used.add(neighbor_idx)
        else:
            faces.extend(f.subdivide())
            used.add(i)

    return faces

# --------------------------
# Relaxation
# --------------------------

def mark_boundary_vertices(faces):
    edge_count = {}
    for face in faces:
        v = face.vertices
        n = len(v)
        for i in range(n):
            a,b = v[i], v[(i+1)%n]
            key = tuple(sorted([id(a),id(b)]))
            edge_count.setdefault(key, []).append((a,b))
    boundary_vertices = set()
    for edge_key, pairs in edge_count.items():
        if len(pairs)==1:
            boundary_vertices.update(pairs[0])
    for vertex in boundary_vertices:
        vertex.is_outer=True

def relax_laplacian(faces, iterations=1, strength=0.5):
    mark_boundary_vertices(faces)
    vertex_neighbors = {}
    for face in faces:
        verts = face.vertices
        n = len(verts)
        for i in range(n):
            v1,v2 = verts[i], verts[(i+1)%n]
            vertex_neighbors.setdefault(v1,set()).add(v2)
            vertex_neighbors.setdefault(v2,set()).add(v1)
    for _ in range(iterations):
        new_pos = {}
        for v, neighbors in vertex_neighbors.items():
            if v.is_outer: continue
            avg_x = sum(n.point[0] for n in neighbors)/len(neighbors)
            avg_y = sum(n.point[1] for n in neighbors)/len(neighbors)
            new_pos[v] = (v.point[0]+(avg_x-v.point[0])*strength,
                          v.point[1]+(avg_y-v.point[1])*strength)
        for v,p in new_pos.items():
            old_key = v.point
            if old_key in Vertex._vertices: del Vertex._vertices[old_key]
            v.point = p
            Vertex._vertices[Vertex.key(p)] = v

def update_points_from_vertices():
    return [list(v.point) for v in Vertex._vertices.values()]

# --------------------------
# Main
# --------------------------

ring_count = 7
ring_radius = 1.5
points = generate_hex_grid_vertices(ring_count, ring_radius)
triangles = triangulate_hex_grid_vertices(ring_count, points)
faces = merge_and_subdivide(triangles, points)
relax_laplacian(faces, iterations=3, strength=0.4)
points = update_points_from_vertices()

# --------------------------
# Visualization
# --------------------------

patches = []
for face in faces:
    coords = [v.point for v in face.vertices]
    n = len(coords)
    for i in range(n):
        x0,y0 = coords[i]
        x1,y1 = coords[(i+1)%n]
        patches.append([[x0,y0],[x1,y1]])

fig,ax = plt.subplots(figsize=(6,6))
for seg in patches:
    ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color='green', linewidth=2)

x = [p[0] for p in points]
y = [p[1] for p in points]
ax.scatter(x,y,color='blue',zorder=10,s=10)
ax.autoscale()
ax.set_aspect('equal')
plt.axis('off')
plt.show()
