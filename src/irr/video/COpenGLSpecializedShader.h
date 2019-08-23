#ifndef __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "COpenGLExtensionHandler.h"
#include "irr/video/COpenGLShader.h"

namespace irr {
namespace asset
{
class IGLSLCompiler;
}

namespace video
{

class COpenGLSpecializedShader : public IGPUSpecializedShader
{
public:
    COpenGLSpecializedShader(const video::IVideoDriver* _driver, const asset::ICPUBuffer* _spirv, const asset::ISpecializationInfo* _specInfo);


    GLuint getOpenGLName() const { return m_GLname; }

private:
    GLuint m_GLname;
};

}}


#endif//__IRR_C_OPENGL_SPECIALIZED_SHADER_H_INCLUDED__
