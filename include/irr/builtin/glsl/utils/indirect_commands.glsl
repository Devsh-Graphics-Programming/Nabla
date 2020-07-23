struct irr_glsl_DrawArraysIndirectCommand_t
{
    uint  count;
    uint  instanceCount;
    uint  first;
    uint  baseInstance;
};

struct irr_glsl_DrawElementsIndirectCommand_t
{
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};