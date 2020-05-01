struct DrawElementsIndirectCommand_t
{
#ifdef __cplusplus
    #define uint uint32_t
#endif
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
#ifdef __cplusplus
    #undef uint
#endif
};
