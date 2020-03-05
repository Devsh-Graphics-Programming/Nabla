#include "optix.h"

#include "stdio.h"

#include "common.h"

extern "C" {
    __constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    if (launch_index.x == 0 && launch_index.y == 0)
        printf("GPU %p\n",rtData);
    params.image[launch_index.y * params.image_width + launch_index.x] = make_uchar4(127,0,0, 255);
}
