"""Writing .cmesh files (flux/robot_mesh.h), shared by the robot asset tools."""
import struct

import numpy as np


def smooth_normals(tri, crease_deg=35.0):
    """Per-corner normals: the area-weighted mean of the normals of the faces at the
    same position whose normal is within crease_deg of this face's."""
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 * area
    unit = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    corners = tri.reshape(-1, 3)
    face = np.repeat(np.arange(len(tri)), 3)
    _, key = np.unique(np.round(corners / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    key = key.ravel()
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    cos_t = np.cos(np.radians(crease_deg))
    normals = np.repeat(unit, 3, 0)
    for grp in np.split(order, bounds):
        if len(grp) < 2:
            continue
        f = face[grp]
        U = unit[f]
        W = (U @ U.T >= cos_t).astype(np.float64)
        N = W @ fn[f]
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)
        normals[grp] = N
    return normals.reshape(-1, 3, 3)


def index_corners(tri, normals, col, link):
    """Unique vertices (position, normal, color, link) and the triangle indices into them."""
    vdt = np.dtype([("p", "<3f4"), ("n", "<3f4"), ("c", "u1", 4), ("l", "<u4")])
    v = np.zeros(len(tri) * 3, vdt)
    v["p"] = tri.reshape(-1, 3)
    v["n"] = normals.reshape(-1, 3)
    v["c"][:, :3] = np.repeat(col, 3, 0)
    v["c"][:, 3] = 255
    v["l"] = link
    uniq, inv = np.unique(v.view(np.dtype((np.void, vdt.itemsize))), return_inverse=True)
    return uniq.view(vdt).ravel(), inv.ravel().astype(np.uint32)


def write_cmesh(path, names, per_link_lods):
    """flux/robot_mesh.h format CRUMESH2: 'CRUMESH2', uint32 raw size, then zlib of:
    nlinks, names (uint32 length + bytes), nlods; per LOD: nverts, nindices, vertices
    {float pos[3], float normal[3], uint8 rgba[4], uint32 link}, uint32 indices (3 per
    triangle). Smooth normals (see smooth_normals); vertices shared between triangles."""
    import zlib
    body = bytearray(struct.pack("<I", len(names)))
    for n in names:
        b = n.encode()
        body += struct.pack("<I", len(b)) + b
    nlods = len(per_link_lods[0])
    body += struct.pack("<I", nlods)
    for lod in range(nlods):
        verts, idx, base = [], [], 0
        for li, lods in enumerate(per_link_lods):
            tri, col = lods[lod]
            if len(tri) == 0:
                continue
            v, i = index_corners(tri, smooth_normals(tri), col, li)
            verts.append(v)
            idx.append(i + base)
            base += len(v)
        allv, alli = np.concatenate(verts), np.concatenate(idx)
        body += struct.pack("<II", len(allv), len(alli)) + allv.tobytes() + alli.tobytes()
    out = b"CRUMESH2" + struct.pack("<I", len(body)) + zlib.compress(bytes(body), 6)
    open(path, "wb").write(out)
