#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

layout(set=0, binding=0) coherent buffer LineCount
{
  DrawIndirectArrays_t lineDraw;
};
layout(set=0, binding=1) writeonly buffer Lines
{
  float linePoints[]; // 6 floats decribe a line, 3d start, 3d end
};

void main() {    
    int i;
    vec3 end = vec3(0,0,0);
    vec3 start = vec3(gl_in[0].gl_Position)
    for(i = 0;i < gl_in.length();i++)
    {
        frag.normal = vertices[i].normal;
        frag.color = vertices[i].color;
        gl_Position = gl_in[i].gl_Position;
        end += vec3(gl_in[i].gl_Position)
        EmitVertex();
    }
    EndPrimitive();

    end /= gl_in.length();
    uint outId = atomicAdd(lineDraw,1u);
    outId *= 6u;
    linePoints[outId+0u] = start.x;
    linePoints[outId+1u] = start.y;
    linePoints[outId+2u] = start.z;
    linePoints[outId+3u] = end.x;
    linePoints[outId+4u] = end.y;
    linePoints[outId+5u] = end.z;

} 