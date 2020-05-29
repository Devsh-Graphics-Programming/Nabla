#ifdef __cplusplus
    #define uint uint32_t
    #define mat4 core::matrix4SIMD
    #define mat3 core::matrix3x4SIMD
#endif

struct DrawElementsIndirectCommand_t
{
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};

struct CullShaderData_t
{
    mat4 viewProjMatrix;
    mat3 viewInverseTransposeMatrix;
    uint maxDrawCount;
    uint cull;
};


#ifdef __cplusplus
    #undef uint
    #undef mat4
    #undef mat3
#endif

#define kCullWorkgroupSize 256