#include "optix.h"

#include "stdint.h"
#include "stdio.h"

#include "common.h"

extern "C" {
    __constant__ Params params;
}

static __forceinline__ __device__ uint64_t irrGetSbtDataPointer()
{
    uint64_t ptr;
    asm("call (%0), _optix_get_sbt_data_ptr_64, ();" : "=l"(ptr) : );
    return ptr;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)irrGetSbtDataPointer();
    if (launch_index.x == 0 && launch_index.y == 0)
        printf("GPU %p\n",rtData);
    params.image[launch_index.y * params.image_width + launch_index.x] = make_uchar4(rtData->r * 255, rtData->g * 255, rtData->b * 255, 255);
}
