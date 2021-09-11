#ifndef _NBL_GLSL_TRANSFORM_TREE_MODIFICATION_REQUEST_RANGE_GLSL_INCLUDED_
#define _NBL_GLSL_TRANSFORM_TREE_MODIFICATION_REQUEST_RANGE_GLSL_INCLUDED_

struct nbl_glsl_transform_tree_modification_request_range_t
{
    uint nodeID;
    int requestsBegin;
    int requestsEnd;
    uint newTimestamp;
};


#endif