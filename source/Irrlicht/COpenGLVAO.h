#ifndef __C_OPEN_GL_VAO_H_INCLUDED__
#define __C_OPEN_GL_VAO_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLBuffer.h"
#include "IMeshBuffer.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{
    static GLint eComponentsPerAttributeToGLint[scene::ECPA_COUNT] =
    {
        GL_BGRA,
        1,2,3,4
    };

    static GLenum eComponentTypeToGLenum[scene::ECT_COUNT] =
    {
        GL_FLOAT,
        GL_HALF_FLOAT,
        GL_DOUBLE,
        GL_UNSIGNED_INT_10F_11F_11F_REV,
        //normalized ints
        GL_INT_2_10_10_10_REV,
        GL_UNSIGNED_INT_2_10_10_10_REV,
        GL_BYTE,
        GL_UNSIGNED_BYTE,
        GL_SHORT,
        GL_UNSIGNED_SHORT,
        GL_INT,
        GL_UNSIGNED_INT,
        //unnorm ints
        GL_INT_2_10_10_10_REV,
        GL_UNSIGNED_INT_2_10_10_10_REV,
        GL_BYTE,
        GL_UNSIGNED_BYTE,
        GL_SHORT,
        GL_UNSIGNED_SHORT,
        GL_INT,
        GL_UNSIGNED_INT,
        //native ints
        GL_INT_2_10_10_10_REV,
        GL_UNSIGNED_INT_2_10_10_10_REV,
        GL_BYTE,
        GL_UNSIGNED_BYTE,
        GL_SHORT,
        GL_UNSIGNED_SHORT,
        GL_INT,
        GL_UNSIGNED_INT,
        //native double
        GL_DOUBLE
    };


    class COpenGLVAO : public scene::IGPUMeshDataFormatDesc
    {
            GLuint vao;
            uint64_t lastValidated;
        public:
            COpenGLVAO();
            virtual ~COpenGLVAO();


            virtual void mapIndexBuffer(IGPUBuffer* ixbuf);

            virtual void mapVertexAttrBuffer(IGPUBuffer* attrBuf, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, scene::E_COMPONENTS_PER_ATTRIBUTE components, scene::E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0);


            inline const GLuint& getOpenGLName() const {return vao;}
            bool rebindRevalidate();
    };


} // end namespace video
} // end namespace irr

#endif
#endif

