
{
    const ivec2 coord = ivec2(gl_FragCoord.xy);

    vec4 fragColor = vec4(color,d);
    float fragDepth = gl_FragCoord.z;

    beginInvocationInterlockARB();

    uint not_empty = imageAtomicOr(g_counter, coord, 1u);

    if (not_empty == 0u)
    {
        oit_per_pixel_data_t data;
        for (int i = 0; i < OIT_VEC4_COUNT; ++i)
        {
            data.color[i] = uvec4(0u);
            data.depth[i] = vec4(FLT_MAX);
        }

        vec4 cf = 255.0*vec4(fragColor.xyz*fragColor.w, 1.0-fragColor.w) + 0.4;
        uint c = uint(cf.x);
        c |= (uint(cf.y) << 8);
        c |= (uint(cf.z) << 16);
        c |= (uint(cf.w) << 24);
        
        data.color[0][0] = c;
        data.depth[0][0] = fragDepth;

        uint addr = oit_genAddr(uvec2(coord));
        g_color.data[addr].color = data.color;
        g_depth.data[addr].depth = data.depth;
    }
    else
    {
        oit_node_t nodes[OIT_NODE_COUNT];

        oit_loadData(uvec2(coord), nodes);

        oit_insertFragment(fragDepth, 1.0-fragColor.w, fragColor.xyz, nodes);

        oit_storeData(uvec2(coord), nodes);
    }

    endInvocationInterlockARB();
}


{
    const ivec2 coord = ivec2(gl_FragCoord.xy);

    vec4 fragColor = vec4(color,d);
    float fragDepth = gl_FragCoord.z;


    const uint real_transparency_layers = imageAtomicAdd(g_counter, coord, 1u);

    if (real_transparency_layers<OIT_NODE_COUNT)
    {
        g_color.data[oit_genAddr(uvec2(coord))].color[real_transparency_layers] = data.color;
        g_depth.data[oit_genAddr(uvec2(coord))].depth[real_transparency_layers] = data.depth;
    }
    else
    {
        oit_node_t nodes[OIT_NODE_COUNT];
        
        beginInvocationInterlockARB(); // TODO: find out about branching on interlocks
        oit_loadData(uvec2(coord), nodes);

        oit_insertFragment(fragDepth, 1.0-fragColor.w, fragColor.xyz, nodes);

        oit_storeData(uvec2(coord), nodes);
        endInvocationInterlockARB();
    }
}
