#ifndef _NBL_GLSL_PROPERTY_POOL_TRANSFER_GLSL_INCLUDED_
#define _NBL_GLSL_PROPERTY_POOL_TRANSFER_GLSL_INCLUDED_

struct nbl_glsl_property_pool_transfer_t
{
	int propertyDWORDsize_upDownFlag;
    int elementCount;
    int indexOffset;
};
#define _NBL_BUILTIN_PROPERTY_POOL_TRANSFER_T_SIZE 12

#endif