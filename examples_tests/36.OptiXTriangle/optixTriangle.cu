#include "irr/builtin/optix/workarounds.h"
#include "optix.h"

#include "common.h"

extern "C" {
    __constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)irrGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] = make_uchar4(rtData->r * 255, rtData->g * 255, rtData->b * 255, 255);
}
