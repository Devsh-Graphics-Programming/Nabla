#include "shaderCommon.glsl"



// TODO: test the fallbacks later
#define _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ ((_IRR_GLSL_WORKGROUP_SIZE_<<2)+4)
shared uint tmpShared[_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_];
#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ tmpShared



#include "irr/builtin/glsl/workgroup/arithmetic.glsl"