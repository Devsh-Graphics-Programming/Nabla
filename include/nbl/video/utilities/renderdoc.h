#ifndef __NBL_VIDEO_UTILITIES_RENDERDOC_H_INCLUDED_
#define __NBL_VIDEO_UTILITIES_RENDERDOC_H_INCLUDED_

#include "renderdoc/renderdoc_app.h" // renderdoc_app from /3rdparty

namespace nbl::video 
{
    using renderdoc_api_t = RENDERDOC_API_1_4_1;
    constexpr static inline auto MinRenderdocVersion = eRENDERDOC_API_Version_1_4_1;
}

#endif //__NBL_VIDEO_UTILITIES_RENDERDOC_H_INCLUDED_