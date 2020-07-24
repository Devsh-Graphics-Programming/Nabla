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

struct irr_glsl_DispatchIndirectCommand_t
{
    uint  num_groups_x;
    uint  num_groups_y;
    uint  num_groups_z;
};