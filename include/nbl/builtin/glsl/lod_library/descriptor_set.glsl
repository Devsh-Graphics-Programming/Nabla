#ifndef _NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET_GLSL_INCLUDED_
#define _NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET_GLSL_INCLUDED_


#ifndef NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET
#define NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET 0
#endif

#ifndef NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_BINDING
#define NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_BINDING 0
#ifndef NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_QUALIFIERS
#define NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
layout(
    set=NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET,
    binding=NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_BINDING
) NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_QUALIFIERS buffer LodTables
{
    uvec4 data[];
} lodTables;

uint nbl_glsl_lod_library_Table_getLoDCount(in uint lodTableUvec4Offset)
{
    const uint offsetofLevelCount = 3u;
    return lodTables.data[lodTableUvec4Offset][offsetofLevelCount];
}
mat2x3 nbl_glsl_lod_library_Table_getAABB(in uint lodTableUvec4Offset)
{
    const uint uvec4OffsetofMaxAABB = 1u;
    return mat2x3(uintBitsToFloat(lodTables.data[lodTableUvec4Offset].xyz),uintBitsToFloat(lodTables.data[lodTableUvec4Offset+uvec4OffsetofMaxAABB].xyz));
}
uint nbl_glsl_lod_library_Table_getLoDUvec4Offset(in uint lodTableUvec4Offset, in uint lodID)
{
    const uint offsetofFirstLoDUvec4Offset = 7u;
    const uint offsetofLodUvec4Offset = lodID+offsetofFirstLoDUvec4Offset;
    return lodTables.data[lodTableUvec4Offset+(offsetofLodUvec4Offset>>2u)][offsetofLodUvec4Offset&0x3u];
}
#endif


#ifndef NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_BINDING
#define NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_BINDING 1
#ifndef NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_QUALIFIERS
#define NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
layout(
    set=NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET,
    binding=NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_BINDING
) NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_QUALIFIERS buffer LodInfos
{
    uint data[];
} lodInfos;

uint nbl_glsl_lod_library_Info_getDrawcallInfoCount(in uint lodInfoUvec4Offset)
{
    const uint lodInfoDWORDOffset = lodInfoUvec4Offset<<2u;
    const uint offsetof_drawcallInfoCountAndTotalBoneCount = 3u;
    return lodInfos.data[lodInfoDWORDOffset+offsetof_drawcallInfoCountAndTotalBoneCount]&0xffffu;
}
uint nbl_glsl_lod_library_Info_getTotalBoneCount(in uint lodInfoUvec4Offset)
{
    const uint lodInfoDWORDOffset = lodInfoUvec4Offset<<2u;
    const uint offsetof_drawcallInfoCountAndTotalBoneCount = 3u;
    return lodInfos.data[lodInfoDWORDOffset+offsetof_drawcallInfoCountAndTotalBoneCount]>>16u;
}
/*
mat2x3 nbl_glsl_lod_library_Info_getAABB(in uint lodInfoUvec4Offset)
{
    return mat2x3(
    );
}
*/
uint nbl_glsl_lod_library_getDrawCallDWORDOffset(in uint lodInfoUvec4Offset, in uint offsetofDWORDFirstDrawcallInfo, in uint drawcallID)
{
    return lodInfos.data[(lodInfoUvec4Offset<<2u)+offsetofDWORDFirstDrawcallInfo+(drawcallID<<1u)];
}
#endif


#endif