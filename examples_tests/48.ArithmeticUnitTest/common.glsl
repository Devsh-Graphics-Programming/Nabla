#include "irr/builtin/glsl/workgroup/arithmetic.glsl"

layout(set = 0, binding = 0) readonly buffer inputBuff
{
    float inputValue[]; 
};
layout(set = 0, binding = 1) writeonly buffer outand
{
    float outputValue[];
};
layout(set = 0, binding = 2) writeonly buffer outxor
{
    float outputValue[];
};
layout(set = 0, binding = 3) writeonly buffer outor
{
    float outputValue[];
};
layout(set = 0, binding = 4) writeonly buffer outadd
{
    float outputValue[];
};
layout(set = 0, binding = 5) writeonly buffer outmul
{
    float outputValue[];
};
layout(set = 0, binding = 6) writeonly buffer outmin
{
    float outputValue[];
};
layout(set = 0, binding = 7) writeonly buffer outmax
{
    float outputValue[];
};