#include "shaderCommon.glsl"

// ORDER OF INCLUDES MATTERS !!!!!
// first the feature that requires the most shared memory should be included
// anyway when one is using more than 2 features that rely on shared memory,
// they should declare the shared memory of appropriate size by themselves.
// But in this unit test we don't because we need to test if the default
// sizing macros actually work for all workgroup sizes.
#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>
#include <nbl/builtin/glsl/workgroup/ballot.glsl>