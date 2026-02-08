import sys

def parse_obj(obj_path, out_path):
    uvs = []
    vertex_uvs = {}
    triangles = []

    max_v_idx = -1

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('vt '):
                parts = line.strip().split()
                u = float(parts[1])
                v = float(parts[2])
                uvs.append([u, v])
            elif line.startswith('f '):
                parts = line.strip().split()

                

                face_verts = []
                for p in parts[1:]:
                    sub = p.split('/')
                    v_idx = int(sub[0]) - 1
                    vt_idx = int(sub[1]) - 1
                    
                    face_verts.append(v_idx)
                    
                    if v_idx not in vertex_uvs:
                        vertex_uvs[v_idx] = uvs[vt_idx]
                    
                    if v_idx > max_v_idx:
                        max_v_idx = v_idx
                

                v0 = face_verts[0]
                for i in range(1, len(face_verts) - 1):
                    v1 = face_verts[i]
                    v2 = face_verts[i+1]
                    triangles.extend([v0, v1, v2])

    print(f"Max vertex index: {max_v_idx}")
    print(f"Found {len(uvs)} UVs")
    print(f"Parsed {len(triangles)//3} triangles")


    final_uvs = []
    

    
    expected_verts = 468
    
    js_content = "export const CANONICAL_TRIANGLES = [\n"
    js_content += ",".join(map(str, triangles))
    js_content += "];\n\n"
    
    js_content += "export const CANONICAL_UVS = [\n"
    for i in range(expected_verts):
        if i in vertex_uvs:
            u, v = vertex_uvs[i]
            js_content += f"  {{ x: {u}, y: {1-v} }},\n"
        else:
            js_content += "  { x: 0, y: 0 },\n"
    js_content += "];\n"

    with open(out_path, 'w') as f:
        f.write(js_content)

if __name__ == "__main__":
    parse_obj("canonical_face.obj", "src/face_mesh_data.js")
