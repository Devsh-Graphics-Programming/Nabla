#ifndef _OIT_GLSL_INCLUDED_
#define _OIT_GLSL_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl>

#ifdef __cplusplus
#define vec4 nbl::core::vectorSIMDf
#define uvec4 nbl::core::vectorSIMDu32
#endif

#define OIT_NODE_COUNT 32
#define OIT_VEC4_COUNT OIT_NODE_COUNT/4

// TODO: RGB9E5 for color (Dirt, etc), 24bit depth and 8bit alpha/visibility
struct oit_per_pixel_data_t
{
    vec4  depth[OIT_VEC4_COUNT];
    uvec4 color[OIT_VEC4_COUNT];
};

struct oit_per_pixel_color_t
{
    uvec4 color[OIT_VEC4_COUNT];
};

struct oit_per_pixel_depth_t
{
    vec4 depth[OIT_VEC4_COUNT];
};

#ifndef __cplusplus

#if 0
layout (push_constant) uniform PushConst {
    uvec2 scr_res;
} PC;
#endif

uvec2 oit_getScrRes() 
{
#if 0
    return PC.scr_res;
#else
    return uvec2(1280u,720u);
#endif
}

layout (set = 2, binding = 0, std430) coherent buffer Color
{
    oit_per_pixel_color_t data[];
} g_color;

layout (set = 2, binding = 1, std430) coherent buffer Depth
{
    oit_per_pixel_depth_t data[];
} g_depth;

layout (set = 2, binding = 2, r32ui) uniform uimage2D g_counter;

uint oit_genAddr(uvec2 addr2d)
{
    uint w = oit_getScrRes().x;
#if 0
    w >>= 1u;
    uvec2 tile_addr_2d = addr2d >> 1u;
    uint tile_addr = (tile_addr_2d.x + w*tile_addr_2d.y) << 2u;
    uvec2 px_addr_2d = addr2d & 1u;
    uint px_addr = (px_addr_2d.y << 1u) | px_addr_2d.x;

    return (tile_addr | px_addr);
#else
    return w*addr2d.y + addr2d.x;
#endif
}

struct oit_node_t
{
    float depth;
    float trans;
    uint color;
};

void oit_loadData(in uvec2 addr2d, out oit_node_t[OIT_NODE_COUNT] out_nodes)
{
    oit_per_pixel_data_t data;
    const uint addr = oit_genAddr(addr2d);

    data.color = g_color.data[addr].color;
    data.depth = g_depth.data[addr].depth;

    for (uint i = 0u; i < OIT_VEC4_COUNT; ++i)
    {
        for (uint j = 0u; j < 4u; ++j)
        {
            const uint lc_addr = 4u*i + j;

            float d = data.depth[i][j];
            float t = float(data.color[i][j] >> 24u) / 255.0;

            out_nodes[lc_addr].depth = d;
            out_nodes[lc_addr].trans = t;
            out_nodes[lc_addr].color = data.color[i][j];
        }
    }
}

void oit_storeData(in uvec2 addr2d, in oit_node_t[OIT_NODE_COUNT] in_nodes)
{
    oit_per_pixel_data_t data;
    const uint addr = oit_genAddr(addr2d);

    for (uint i = 0u; i < OIT_VEC4_COUNT; ++i)
    {
        for (uint j = 0u; j < 4u; ++j)
        {
            const uint lc_addr = 4u*i + j;

            uint color = in_nodes[lc_addr].color;
            color &= 0x00FFFFFFu;
            float t = in_nodes[lc_addr].trans;
            color |= ((uint(t*255.0 + 0.4)) << 24u);
            data.depth[i][j] = in_nodes[lc_addr].depth;
            data.color[i][j] = color;
        }
    }

    g_color.data[addr].color = data.color;
    g_depth.data[addr].depth = data.depth;
}

void oit_insertFragment(in float fragDepth, in float fragTrans, in vec3 fragColor, inout oit_node_t[OIT_NODE_COUNT] nodes)
{
    float depth[OIT_NODE_COUNT + 1];
    float trans[OIT_NODE_COUNT + 1];
    uint  color[OIT_NODE_COUNT + 1];

    for (int i = 0; i < OIT_NODE_COUNT; ++i)
    {
        depth[i] = nodes[i].depth;
        trans[i] = nodes[i].trans;
        color[i] = nodes[i].color;
    }

    int idx = 0;
    float prevTrans = 1.0;
    for (int i = 0; i < OIT_NODE_COUNT; ++i)
    {
        if (fragDepth > depth[i])
        {
            idx++;
            prevTrans = trans[i];
        }
    }

    for (int i = OIT_NODE_COUNT - 1; i >= 0; --i)
    {
        if (i >= idx)
        {
            depth[i+1] = depth[i];
            trans[i+1] = prevTrans*trans[i];
            color[i+1] = color[i];
        }
    }

    const float newTrans = fragTrans * prevTrans;
    vec3 newColorf = fragColor * (1.0 - fragTrans); // ???
    newColorf *= 255.0;
    newColorf += 0.4;
    uint newColor = uint(newColorf.r);
    newColor |= (uint(newColorf.g) << 8);
    newColor |= (uint(newColorf.b) << 16);

    depth[idx] = fragDepth;
    trans[idx] = newTrans;
    color[idx] = newColor;

    if (depth[OIT_NODE_COUNT] < FLT_MAX) //overflow
    {
        uint c = color[OIT_NODE_COUNT];
        vec3 rm = vec3(float(c&0xffu), float((c>>8)&0xffu), float((c>>16)&0xffu));
        c = color[OIT_NODE_COUNT - 1];
        vec3 acc = vec3(float(c&0xffu), float((c>>8)&0xffu), float((c>>16)&0xffu));
        
        vec3 cf = acc + (rm * trans[OIT_NODE_COUNT-1] / trans[OIT_NODE_COUNT-2]);
        c = uint(cf.x);
        c |= (uint(cf.y) << 8);
        c |= (uint(cf.z) << 16);
        color[OIT_NODE_COUNT-1] = c;
        trans[OIT_NODE_COUNT-1] = trans[OIT_NODE_COUNT];
    }

    for (int i = 0; i < OIT_NODE_COUNT; ++i)
    {
        nodes[i].depth = depth[i];
        nodes[i].trans = trans[i];
        nodes[i].color = color[i];
    }
}

#endif //!__cplusplus

#endif //_OIT_GLSL_INCLUDED_
