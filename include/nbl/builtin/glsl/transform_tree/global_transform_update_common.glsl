layout(local_size_x=NBL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS) in;

#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_RECOMPUTED_TIMESTAMP_DESCRIPTOR_QUALIFIERS coherent restrict
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_GLOBAL_TRANSFORM_DESCRIPTOR_QUALIFIERS coherent restrict
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_NORMAL_MATRIX_DESCRIPTOR_QUALIFIERS writeonly restrict
#include "nbl/builtin/glsl/transform_tree/pool_descriptor_set.glsl"
#include "nbl/builtin/glsl/transform_tree/global_transform_update_descriptor_set.glsl"

#include "nbl/builtin/glsl/utils/transform.glsl"
#include "nbl/builtin/glsl/utils/normal_encode.glsl"
void nbl_glsl_transform_tree_globalTransformUpdate_updateNormalMatrix(in uint nodeID, in mat4x3 globalTransform);

// TODO: figure out which `memoryBarrierBuffer();` can be removed

// there isn't any lock so there's no mutual exclusion here, we can, and we will end up doing duplicate work
void nbl_glsl_transform_tree_globalTransformUpdate_impl(in uint nodeID, in mat4x3 updatedTransform, in uint expectedTimestamp)
{
    memoryBarrierBuffer();
    // these data races are completely fine, all threads will write the same value
    nodeGlobalTransforms.data[nodeID] = updatedTransform;
    memoryBarrierBuffer();
    // timestamp needs to be updated AFTER global transform is written
    nodeRecomputedTimestamp.data[nodeID] = expectedTimestamp;

    // doesn't matter when this gets written, and doesnt matter if coherently
    nbl_glsl_transform_tree_globalTransformUpdate_updateNormalMatrix(nodeID,updatedTransform);
}

mat4x3 nbl_glsl_transform_tree_globalTransformUpdate_root(in uint nodeID)
{
    const uint expectedTimestamp = nodeUpdatedTimestamp.data[nodeID];
    const mat4x3 retval = nodeRelativeTransforms.data[nodeID];
    memoryBarrierBuffer();
    // not updated yet, update
    if (nodeRecomputedTimestamp.data[nodeID]!=expectedTimestamp)
        nbl_glsl_transform_tree_globalTransformUpdate_impl(nodeID,retval,expectedTimestamp);
    // relative transform == global transform for a root node
    return retval;
}
void nbl_glsl_transform_tree_globalTransformUpdate_child(inout mat4x3 accumulatedTransform, in uint nodeID)
{
    const uint expectedTimestamp = nodeUpdatedTimestamp.data[nodeID];
    memoryBarrierBuffer();
    // already updated fully, global transform is safe to read
    if (nodeRecomputedTimestamp.data[nodeID]==expectedTimestamp)
    {
        memoryBarrierBuffer();
        accumulatedTransform = nodeGlobalTransforms.data[nodeID];
        return;
    }
    //
    memoryBarrierBuffer();
    accumulatedTransform = nbl_glsl_pseudoMul4x3with4x3(accumulatedTransform,nodeRelativeTransforms.data[nodeID]);
    nbl_glsl_transform_tree_globalTransformUpdate_impl(nodeID,accumulatedTransform,expectedTimestamp);
}


#include "nbl/builtin/glsl/property_pool/transfer.glsl"

// TODO: move and find a way to verify the depths in the TransformTree
#define NBL_GLSL_TRANSFORM_TREE_MAX_DEPTH 13

void main()
{
    // in the future, maybe put it in shared mem or something
#define NBL_GLSL_TRANSFORM_TREE_STACK_SIZE (NBL_GLSL_TRANSFORM_TREE_MAX_DEPTH-1)
    uint stack[NBL_GLSL_TRANSFORM_TREE_STACK_SIZE];

    const uint dispatchSize = NBL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS*gl_NumWorkGroups[0];
    for (uint nodeID=gl_GlobalInvocationID.x; nodeID<nodesToUpdate.count; nodeID+=dispatchSize)
    {
        int stackPtr = 0;

        uint rootNode = nodesToUpdate.data[nodeID];
        uint parent = nodeParents.data[rootNode];
        for (int i=0; i<NBL_GLSL_TRANSFORM_TREE_STACK_SIZE && parent!=NBL_GLSL_PROPERTY_POOL_INVALID; i++)
        {
            stack[stackPtr++] = rootNode;
            rootNode = parent;
            parent = nodeParents.data[rootNode]; 
        }
   
        //
        mat4x3 accumulatedTransform = nbl_glsl_transform_tree_globalTransformUpdate_root(rootNode);
        for (int i=0; i<NBL_GLSL_TRANSFORM_TREE_STACK_SIZE && bool(stackPtr--); i++)
#undef NBL_GLSL_TRANSFORM_TREE_STACK_SIZE
        {
            // barriers are needed to ensure the timestamp and global transform writes are made visible
            // **in the correct order** (global transform must be visible before timestamp) every time we check them.
            // This ensures that if the timestamp matches, the global transform is safe to read (contains updated values).
            memoryBarrierBuffer();
            nbl_glsl_transform_tree_globalTransformUpdate_child(accumulatedTransform,stack[stackPtr]);
        }
    }
}