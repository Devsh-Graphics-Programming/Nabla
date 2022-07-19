#ifndef _NBL_GLSL_TRANSFORM_TREE_RELATIVE_TRANSFORM_UPDATE_COMMON_GLSL_INCLUDED_
#define _NBL_GLSL_TRANSFORM_TREE_RELATIVE_TRANSFORM_UPDATE_COMMON_GLSL_INCLUDED_

#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_RELATIVE_TRANSFORM_DESCRIPTOR_QUALIFIERS restrict
#include "nbl/builtin/glsl/transform_tree/pool_descriptor_set.glsl"

#include "nbl/builtin/glsl/transform_tree/relative_transform_update_descriptor_set.glsl"

void nbl_glsl_transform_tree_relativeTransformUpdate_noStamp(in nbl_glsl_transform_tree_modification_request_range_t requestRange)
{
    mat4x3 updatedTransform;
    // slight optimization
    if (nbl_glsl_transform_tree_relative_transform_modification_t_getType(relativeTransformModifications.data[requestRange.requestsBegin])!=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_OVERWRITE_)
        updatedTransform = nodeRelativeTransforms.data[requestRange.nodeID];

    // OpenGL drivers (even Nvidia) have some bugs and can sporadically completely
    // forget about memory or execution barriers between `glCopyBufferSubData` and compute dispatches
    // as well as modify memory to bogus values even though you're always copying the same data to the same location
    const int length = min(requestRange.requestsEnd-requestRange.requestsBegin,relativeTransformModificationRequestRanges.maxRangeLength);
    for (int i=0; i<length; i++)
        updatedTransform = nbl_glsl_transform_tree_relative_transform_modification_t_apply(updatedTransform,relativeTransformModifications.data[requestRange.requestsBegin+i]);

    nodeRelativeTransforms.data[requestRange.nodeID] = updatedTransform;
}

#endif