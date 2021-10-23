#ifndef _NBL_GLSL_PROPERTY_POOL_TRANSFER_GLSL_INCLUDED_
#define _NBL_GLSL_PROPERTY_POOL_TRANSFER_GLSL_INCLUDED_

struct nbl_glsl_property_pool_transfer_t
{
	int propertyDWORDsize_flags;
    int elementCount;
    int srcIndexOffset;
    int dstIndexOffset;
};
#define NBL_BUILTIN_PROPERTY_POOL_TRANSFER_EF_DOWNLOAD 0x1u
#define NBL_BUILTIN_PROPERTY_POOL_TRANSFER_EF_SRC_FILL 0x2u
#define NBL_BUILTIN_PROPERTY_POOL_TRANSFER_EF_BIT_COUNT 2

#define NBL_BUILTIN_PROPERTY_POOL_INVALID 0xdeadbeef

#endif