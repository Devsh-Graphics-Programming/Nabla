#ifndef __C_OPENGLES_FEATURE_MAP_H_INCLUDED__
#define __C_OPENGLES_FEATURE_MAP_H_INCLUDED__

#include <cstdint>

namespace nbl {
namespace video
{

class COpenGLESFeatureMap
{
public:
	//Version of OpenGL multiplied by 100 - 4.4 becomes 440
	uint16_t Version = 0;
	uint16_t ShaderLanguageVersion = 0;
};

}
}

#endif
