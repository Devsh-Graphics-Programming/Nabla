// This is a struct used to communicate launch parameters which are constant
// for all threads in a given optixLaunch call. 
struct Params
{
    uchar4*  image;
    unsigned int  image_width;
    unsigned int  image_height;
    float3   cam_eye;
    float3   cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

extern "C" 
{
__constant__ Params params;
}

// Note the __raygen__ prefix which marks this as a ray-generation
// program function
extern "C" __global__ void __raygen__rg() 
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
 
    // Map our launch idx to a screen location and create a ray from 
    // the camera location through the screen
    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );
 
    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,   // Min intersection distance
        1e16f,  // Max intersection distance
        0.0f,   // ray-time -- used for motion blur
        OptixVisibilityMask( 255 ), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,      // SBT offset -- See SBT discussion
        0,      // SBT stride -- See SBT discussion 
        0,      // missSBTIndex -- See SBT discussion
        p0, p1, p2 ); // These 32b values are the ray payload
 
    // Our results were packed into opaque 32b registers
    float3 result;
    result.x = int_as_float( p0 );
    result.y = int_as_float( p1 );
    result.z = int_as_float( p2 );
 
    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color( result );
}

extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental 
    // attributes are provided by the OptiX API, including barycentric 
    // coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();

    // Convert to color and assign to our payload outputs.
    const float3 c = make_float3(barycentrics, 1.0f);
    optixSetPayload_0(float_as_int(c.x));
    optixSetPayload_1(float_as_int(c.y));
    optixSetPayload_2(float_as_int(c.z));
}

extern "C" __global__ void __miss__ms()
{
    MissData* miss_data =
        reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(miss_data->bg_color);
}