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
            GLuint vao; // delete
            uint64_t lastValidated;

            bool rebindRevalidate();

            core::LeakDebugger* leakDebugger;

        protected:
            virtual ~COpenGLVAO();

        public:
            COpenGLVAO(core::LeakDebugger* dbgr=NULL);


            virtual void mapIndexBuffer(IGPUBuffer* ixbuf);

            virtual void mapVertexAttrBuffer(IGPUBuffer* attrBuf, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, scene::E_COMPONENTS_PER_ATTRIBUTE components, scene::E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0, uint32_t divisor=0);

            virtual void setMappedBufferOffset(const scene::E_VERTEX_ATTRIBUTE_ID& attrId, const size_t &offset);


            virtual const HashAttribs* getHash()
            {
                if (rebindRevalidate())
                    return &individualHashFields;
                else
                    return NULL;
            }

            inline const GLuint& getOpenGLName() const {return vao;}


            void swapVertexAttrBuffer(IGPUBuffer* attrBuf, const scene::E_VERTEX_ATTRIBUTE_ID& attrId, const size_t& newOffset, const size_t& newStride)
            {
                if (!mappedAttrBuf[attrId] || !attrBuf)
                    return;

                attrBuf->grab();
                mappedAttrBuf[attrId]->drop();

                COpenGLBuffer* asGLBuf = dynamic_cast<COpenGLBuffer*>(attrBuf);
                mappedAttrBuf[attrId] = asGLBuf;
                attrOffset[attrId] = newOffset;
                attrStride[attrId] = newStride;

                COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(vao,attrId,asGLBuf->getOpenGLName(),attrOffset[attrId],attrStride[attrId]);
            }
    };


} // end namespace video
} // end namespace irr

#endif
#endif

