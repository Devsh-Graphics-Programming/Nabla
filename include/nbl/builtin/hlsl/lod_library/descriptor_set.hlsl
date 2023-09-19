
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_LOD_LIBRARY_DESCRIPTOR_SET_HLSL_INCLUDED_
#define _NBL_HLSL_LOD_LIBRARY_DESCRIPTOR_SET_HLSL_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace lod_library
{


#include <nbl/builtin/hlsl/shapes/aabb.hlsl>
#include <nbl/builtin/hlsl/common.hlsl>

#ifndef DESCRIPTOR_SET
#define DESCRIPTOR_SET 0
#endif

#ifndef LOD_TABLES_DESCRIPTOR_BINDING
#define LOD_TABLES_DESCRIPTOR_BINDING 0
#endif
#ifndef LOD_TABLES_DESCRIPTOR_QUALIFIERS
#define LOD_TABLES_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
#ifndef LOD_TABLES_DESCRIPTOR_DECLARED
#define LOD_TABLES_DESCRIPTOR_DECLARED

layout(
    set=DESCRIPTOR_SET,
    binding=LOD_TABLES_DESCRIPTOR_BINDING
) LOD_TABLES_DESCRIPTOR_QUALIFIERS buffer LodTables
{
    uvec4 data[];
} lodTables;


uint Table_getLoDCount(in uint lodTableUvec4Offset)
{
    const uint offsetofLevelCount = 3u;
    return lodTables.data[lodTableUvec4Offset][offsetofLevelCount];
}
shapes::AABB_t Table_getAABB(in uint lodTableUvec4Offset)
{
    const uint uvec4OffsetofMaxAABB = 1u;
    return shapes::AABB_t(
        asfloat(lodTables.data[lodTableUvec4Offset].xyz),
        asfloat(lodTables.data[lodTableUvec4Offset+uvec4OffsetofMaxAABB].xyz)
    );
}
uint Table_getLoDUvec2Offset(in uint lodTableUvec4Offset, in uint lodID)
{
    const uint offsetofFirstLoDUvec2Offset = 7u;
    const uint offsetofLodUvec2Offset = lodID+offsetofFirstLoDUvec2Offset;
    return lodTables.data[lodTableUvec4Offset+(offsetofLodUvec2Offset>>2u)][offsetofLodUvec2Offset&0x3u];
}
#endif


#ifndef LOD_INFOS_DESCRIPTOR_BINDING
#define LOD_INFOS_DESCRIPTOR_BINDING 1
#endif
#ifndef LOD_INFOS_DESCRIPTOR_QUALIFIERS
#define LOD_INFOS_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
#ifndef LOD_INFOS_DESCRIPTOR_DECLARED
#define LOD_INFOS_DESCRIPTOR_DECLARED

layout(
    set=DESCRIPTOR_SET,
    binding=LOD_INFOS_DESCRIPTOR_BINDING
) LOD_INFOS_DESCRIPTOR_QUALIFIERS buffer LodInfos
{
    uvec2 data[];
} lodInfos;

uint Info_getDrawcallInfoCount(in uint lodInfoUvec2Offset)
{
    const int bitoffset_drawcallInfoCount = 0;
    return bitfieldExtract(lodInfos.data[lodInfoUvec2Offset][0],bitoffset_drawcallInfoCount,16);
}
uint Info_getTotalBoneCount(in uint lodInfoUvec2Offset)
{
    const int bitoffset_totalBoneCount = 16;
    return bitfieldExtract(lodInfos.data[lodInfoUvec2Offset][0],bitoffset_totalBoneCount,16);
}

#include <nbl/builtin/hlsl/lod_library/structs.hlsl>
#include <nbl/builtin/hlsl/format/decode.hlsl>
shapes::AABB_t Info_getAABB(in uint lodInfoUvec2Offset, in uint offsetofUvec2FirstDrawcallInfo, in uint drawcallID)
{
    const uint uvec2OffsetofMinAABB = 0u;
    const uint uvec2OffsetofMaxAABB = 1u;
    const uint offset = lodInfoUvec2Offset+offsetofUvec2FirstDrawcallInfo+drawcallID * DRAWCALL_INFO_UVEC2_SIZE;
    return shapes::AABB_t(
        decodeRGB18E7S3(lodInfos.data[offset+uvec2OffsetofMinAABB]),
        decodeRGB18E7S3(lodInfos.data[offset+uvec2OffsetofMaxAABB])
    );
}
uint Info_getDrawCallDWORDOffset(in uint lodInfoUvec2Offset, in uint offsetofUvec2FirstDrawcallInfo, in uint drawcallID)
{
    const uint uvec2OffsetofDrawcallDWORDOffset = 2u;
    const uint offset = lodInfoUvec2Offset+offsetofUvec2FirstDrawcallInfo+drawcallID*DRAWCALL_INFO_UVEC2_SIZE;
    return lodInfos.data[offset+uvec2OffsetofDrawcallDWORDOffset][0];
}


DefaultLoDChoiceParams DefaultInfo_getLoDChoiceParams(in uint lodInfoUvec2Offset)
{
    const uint offsetofUvec2_lodChoiceParams = LOD_INFO_BASE_SIZE>>3u;
    return DefaultLoDChoiceParams(asfloat(
        lodInfos.data[lodInfoUvec2Offset+offsetofUvec2_lodChoiceParams][(LOD_INFO_BASE_SIZE>>2)&0x1u]
    ));
}

shapes::AABB_t DefaultInfo_getAABB(in uint lodInfoUvec2Offset, in uint drawcallID)
{
    return Info_getAABB(
        lodInfoUvec2Offset, CULLING_LOD_SELECTION_LOD_INFO_DRAWCALL_LIST_UVEC2_OFFSET, drawcallID
    );
}
uint DefaultInfo_getDrawCallDWORDOffset(in uint lodInfoUvec2Offset, in uint drawcallID)
{
    return Info_getDrawCallDWORDOffset(
        lodInfoUvec2Offset, CULLING_LOD_SELECTION_LOD_INFO_DRAWCALL_LIST_UVEC2_OFFSET, drawcallID
    );
}
#endif


}
}
}
#endif