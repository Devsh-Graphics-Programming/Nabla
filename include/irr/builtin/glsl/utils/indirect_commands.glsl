struct DrawArraysIndirectCommand_t
{
    uint  count;
    uint  instanceCount;
    uint  first;
    uint  baseInstance;
};

struct DrawElementsIndirectCommand_t
{
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};