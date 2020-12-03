#ifndef _INSTANCE_DATA_PER_CAMERA_INCLUDED_
#define _INSTANCE_DATA_PER_CAMERA_INCLUDED_

#include "common.glsl"

struct InstanceDataPerCamera
{
    mat4 MVP;
    mat4x3 NormalMatAndFlags;
};

#endif
