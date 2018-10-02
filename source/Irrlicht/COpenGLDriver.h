// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in Irrlicht.h

#ifndef __C_VIDEO_OPEN_GL_H_INCLUDED__
#define __C_VIDEO_OPEN_GL_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "SIrrCreationParameters.h"

namespace irr
{
	class CIrrDeviceWin32;
	class CIrrDeviceLinux;
	class CIrrDeviceSDL;
	class CIrrDeviceMacOSX;
}

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "CNullDriver.h"
#include "IMaterialRendererServices.h"
// also includes the OpenGL stuff
#include "COpenGLExtensionHandler.h"
#include "COpenGLDriverFence.h"
#include "COpenGLTransformFeedback.h"
#include "COpenGLVAOSpec.h"
#include "COpenCLHandler.h"

#include <map>
#include "FW_Mutex.h"

namespace irr
{

namespace video
{
	class COpenGLTexture;
	class COpenGLFrameBuffer;

	class COpenGLDriver : public CNullDriver, public IMaterialRendererServices, public COpenGLExtensionHandler
	{
    protected:
		//! destructor
		virtual ~COpenGLDriver();

	public:
        struct SAuxContext;

		#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceWin32* device);
		//! inits the windows specific parts of the open gl driver
		bool initDriver(CIrrDeviceWin32* device);
		bool changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceWin32* device);
		#endif

		#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceLinux* device);
		//! inits the GLX specific parts of the open gl driver
		bool initDriver(CIrrDeviceLinux* device, SAuxContext* auxCtxts);
		bool changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceLinux* device);
		#endif

		#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceSDL* device);
		#endif

		#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceMacOSX *device);
		#endif

		//! generic version which overloads the unimplemented versions
		bool changeRenderContext(const SExposedVideoData& videoData, void* device) {return false;}

        bool initAuxContext();
        const SAuxContext* getThreadContext(const std::thread::id& tid=std::this_thread::get_id()) const;
        bool deinitAuxContext();

        virtual uint16_t retrieveDisplayRefreshRate() const override;

		//!
		virtual IGPUBuffer* createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false);

	    virtual scene::IGPUMeshDataFormatDesc* createGPUMeshDataFormatDesc(core::LeakDebugger* dbgr=NULL);

        SGPUMaterial makeGPUMaterialFromCPU(const SCPUMaterial& _cpumat);

	    virtual core::vector<scene::IGPUMesh*> createGPUMeshesFromCPU(const core::vector<scene::ICPUMesh*>& mesh);


        virtual void flushBufferRanges(const uint32_t& memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges);

        virtual void copyBuffer(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, const size_t& readOffset, const size_t& writeOffset, const size_t& length);

		//! clears the zbuffer
		virtual bool beginScene(bool backBuffer=true, bool zBuffer=true,
				SColor color=SColor(255,0,0,0),
				const SExposedVideoData& videoData=SExposedVideoData(),
				core::rect<int32_t>* sourceRect=0);

		//! presents the rendered scene on the screen, returns false if failed
		virtual bool endScene();


		virtual void beginQuery(IQueryObject* query);
		virtual void endQuery(IQueryObject* query);
		virtual void beginQuery(IQueryObject* query, const size_t& index);
		virtual void endQuery(IQueryObject* query, const size_t& index);

        virtual IQueryObject* createPrimitivesGeneratedQuery();
        virtual IQueryObject* createXFormFeedbackPrimitiveQuery();
        virtual IQueryObject* createElapsedTimeQuery();
        virtual IGPUTimestampQuery* createTimestampQuery();


        virtual void drawMeshBuffer(const scene::IGPUMeshBuffer* mb);

		virtual void drawArraysIndirect(const scene::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const scene::E_PRIMITIVE_TYPE& mode,
                                        const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride);
		virtual void drawIndexedIndirect(const scene::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                            const scene::E_PRIMITIVE_TYPE& mode,
                                            const scene::E_INDEX_TYPE& type, const IGPUBuffer* indirectDrawBuff,
                                            const size_t& offset, const size_t& count, const size_t& stride);


		//! queries the features of the driver, returns true if feature is available
		virtual bool queryFeature(const E_DRIVER_FEATURE& feature) const;

		//!
		virtual void issueGPUTextureBarrier() {COpenGLExtensionHandler::extGlTextureBarrier();}


		virtual const video::SGPUMaterial& getCurrentMaterial() const {return Material;}

		//! Sets a material. All 3d drawing functions draw geometry now
		//! using this material.
		//! \param material: Material to be used from now on.
		virtual void setMaterial(const SGPUMaterial& material);

        //! needs to be "deleted" since its not refcounted
        virtual IDriverFence* placeFence(const bool& implicitFlushWaitSameThread=false)
        {
            return new COpenGLDriverFence(implicitFlushWaitSameThread);
        }

		//! \return Returns the name of the video driver. Example: In case of the Direct3D8
		//! driver, it would return "Direct3D8.1".
		virtual const wchar_t* getName() const;

		//! sets a viewport
		virtual void setViewPort(const core::rect<int32_t>& area);

		//! Returns type of video driver
		virtual E_DRIVER_TYPE getDriverType() const;

		//! get color format of the current color buffer
		virtual ECOLOR_FORMAT getColorFormat() const;

		//! Can be called by an IMaterialRenderer to make its work easier.
		virtual void setBasicRenderStates(const SGPUMaterial& material, const SGPUMaterial& lastmaterial,
			bool resetAllRenderstates);


        virtual void setShaderConstant(const void* data, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number=1);


        virtual int32_t addHighLevelShaderMaterial(
            const char* vertexShaderProgram,
            const char* controlShaderProgram,
            const char* evaluationShaderProgram,
            const char* geometryShaderProgram,
            const char* pixelShaderProgram,
            uint32_t patchVertices=3,
            E_MATERIAL_TYPE baseMaterial=video::EMT_SOLID,
            IShaderConstantSetCallBack* callback=0,
            const char** xformFeedbackOutputs = NULL,
            const uint32_t& xformFeedbackOutputCount = 0,
            int32_t userData=0,
            const char* vertexShaderEntryPointName="main",
            const char* controlShaderEntryPointName="main",
            const char* evaluationShaderEntryPointName="main",
            const char* geometryShaderEntryPointName="main",
            const char* pixelShaderEntryPointName="main");

		//! Returns a pointer to the IVideoDriver interface. (Implementation for
		//! IMaterialRendererServices)
		virtual IVideoDriver* getVideoDriver();

		//! Returns the maximum amount of primitives (mostly vertices) which
		//! the device is able to render with one drawIndexedTriangleList
		//! call.
		virtual uint32_t getMaximalIndicesCount() const;

		//! .
        virtual ITexture* addTexture(const ITexture::E_TEXTURE_TYPE& type, const core::vector<CImageData*>& images, const io::path& name, ECOLOR_FORMAT format);

        //!
        virtual IMultisampleTexture* addMultisampleTexture(const IMultisampleTexture::E_MULTISAMPLE_TEXTURE_TYPE& type, const uint32_t& samples, const uint32_t* size, ECOLOR_FORMAT format = ECF_A8R8G8B8, const bool& fixedSampleLocations = false);

		//! A.
        virtual ITextureBufferObject* addTextureBufferObject(IGPUBuffer* buf, const ITextureBufferObject::E_TEXURE_BUFFER_OBJECT_FORMAT& format = ITextureBufferObject::ETBOF_RGBA8, const size_t& offset=0, const size_t& length=0);

        virtual IFrameBuffer* addFrameBuffer();

        //! Remove
        virtual void removeFrameBuffer(IFrameBuffer* framebuf);

        virtual void removeAllFrameBuffers();


		virtual bool setRenderTarget(IFrameBuffer* frameBuffer, bool setNewViewport=true);

		virtual void blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out,
                                        bool copyDepth=true, bool copyStencil=true,
										core::recti srcRect=core::recti(0,0,0,0),
										core::recti dstRect=core::recti(0,0,0,0),
										bool bilinearFilter=false);


		//! Clears the ZBuffer.
		virtual void clearZBuffer(const float &depth=0.0);

		virtual void clearStencilBuffer(const int32_t &stencil);

		virtual void clearZStencilBuffers(const float &depth, const int32_t &stencil);

		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals);

		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals);
		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals);


		virtual ITransformFeedback* createTransformFeedback();

		//!
		virtual void bindTransformFeedback(ITransformFeedback* xformFeedback);

		virtual ITransformFeedback* getBoundTransformFeedback() {return getThreadContext_helper(false,std::this_thread::get_id())->CurrentXFormFeedback;}

        /** Only POINTS, LINES, and TRIANGLES are allowed as capture types.. no strips or fans!
        This issues an implicit call to bindTransformFeedback()
        **/
		virtual void beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const scene::E_PRIMITIVE_TYPE& primType=scene::EPT_POINTS);

		//! A redundant wrapper call to ITransformFeedback::pauseTransformFeedback(), made just for clarity
		virtual void pauseTransformFeedback();

		//! A redundant wrapper call to ITransformFeedback::pauseTransformFeedback(), made just for clarity
		virtual void resumeTransformFeedback();

		virtual void endTransformFeedback();


		//! Enable/disable a clipping plane.
		//! There are at least 6 clipping planes available for the user to set at will.
		//! \param index: The plane index. Must be between 0 and MaxUserClipPlanes.
		//! \param enable: If true, enable the clipping plane else disable it.
		virtual void enableClipPlane(uint32_t index, bool enable);

		//! Enable the 2d override material
		virtual void enableMaterial2D(bool enable=true);

		//! Returns the graphics card vendor name.
		virtual std::string getVendorInfo() {return VendorName;}

		//! Removes a texture from the texture cache and deletes it, freeing lot of memory.
		void removeTexture(ITexture* texture);

		//! sets the needed renderstates
		void setRenderStates3DMode();

		//!
		const size_t& getMaxConcurrentShaderInvocations() const {return maxConcurrentShaderInvocations;}

		//!
		const uint32_t& getMaxShaderComputeUnits() const {return maxShaderComputeUnits;}

		//!
		const size_t& getMaxShaderInvocationsPerALU() const {return maxALUShaderInvocations;}

#ifdef _IRR_COMPILE_WITH_OPENCL_
        const cl_device_id& getOpenCLAssociatedDevice() const {return clDevice;}

        const size_t& getOpenCLAssociatedDeviceID() const {return clDeviceIx;}
        const size_t& getOpenCLAssociatedPlatformID() const {return clPlatformIx;}
#endif // _IRR_COMPILE_WITH_OPENCL_

        struct SAuxContext
        {
        //public:
            constexpr static size_t maxVAOCacheSize = 0x1u<<14; //make this cache configurable

            SAuxContext() : threadId(std::thread::id()), ctx(NULL), XFormFeedbackRunning(false), CurrentXFormFeedback(NULL),
                            CurrentFBO(0), CurrentRendertargetSize(0,0)
            {
                VAOMap.reserve(maxVAOCacheSize);
                CurrentVAO = HashVAOPair(COpenGLVAOSpec::HashAttribs(),NULL);

                for (size_t i=0; i<MATERIAL_MAX_TEXTURES; i++)
                {
                    CurrentSamplerHash[i] = 0xffffffffffffffffuLL;
                }
            }

            inline void setActiveSSBO(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** const buffers, const ptrdiff_t* const offsets, const ptrdiff_t* const sizes)
            {
                shaderStorageBufferObjects.set(first,count,buffers,offsets,sizes);
            }

            inline void setActiveUBO(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** const buffers, const ptrdiff_t* const offsets, const ptrdiff_t* const sizes)
            {
                uniformBufferObjects.set(first,count,buffers,offsets,sizes);
            }

            inline void setActiveIndirectDrawBuffer(const COpenGLBuffer* const buff)
            {
                indirectDraw.set(buff);
            }

            bool setActiveVAO(const COpenGLVAOSpec* const spec, const scene::IGPUMeshBuffer* correctOffsetsForXFormDraw=NULL);

            //! sets the current Texture
            //! Returns whether setting was a success or not.
            bool setActiveTexture(uint32_t stage, video::IVirtualTexture* texture, const video::STextureSamplingParams &sampleParams);

            const GLuint& constructSamplerInCache(const uint64_t &hashVal);

        //private:
            std::thread::id threadId;
            #ifdef _IRR_WINDOWS_API_
                HGLRC ctx;
            #endif
            #ifdef _IRR_COMPILE_WITH_X11_DEVICE_
                GLXContext ctx;
                GLXPbuffer pbuff;
            #endif
            #ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
                AppleMakesAUselessOSWhichHoldsBackTheGamingIndustryAndSabotagesOpenStandards ctx;
            #endif

            bool                        XFormFeedbackRunning; // TODO: delete
            COpenGLTransformFeedback*   CurrentXFormFeedback; //TODO: delete


            //! FBOs
            core::vector<IFrameBuffer*>  FrameBuffers;
            COpenGLFrameBuffer*         CurrentFBO;
            core::dimension2d<uint32_t> CurrentRendertargetSize;


            //! Buffers
            template<GLenum BIND_POINT,size_t BIND_POINTS>
            class BoundIndexedBuffer
            {
                    const COpenGLBuffer* boundBuffer[BIND_POINTS];
                    ptrdiff_t boundOffsets[BIND_POINTS];
                    ptrdiff_t boundSizes[BIND_POINTS];
                    uint64_t lastValidatedBuffer[BIND_POINTS];
                public:
                    BoundIndexedBuffer()
                    {
                        memset(boundBuffer,0,sizeof(boundBuffer));
                        memset(boundOffsets,0,sizeof(boundOffsets));
                        memset(boundSizes,0,sizeof(boundSizes));
                        memset(lastValidatedBuffer,0,sizeof(boundBuffer));
                    }

                    ~BoundIndexedBuffer()
                    {
                        set(0,BIND_POINTS,nullptr,nullptr,nullptr);
                    }

                    void set(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** const buffers, const ptrdiff_t* const offsets, const ptrdiff_t* const sizes);
            };

            //! SSBO
            BoundIndexedBuffer<GL_SHADER_STORAGE_BUFFER,OGL_MAX_BUFFER_BINDINGS>    shaderStorageBufferObjects;
            //! UBO
            BoundIndexedBuffer<GL_UNIFORM_BUFFER,OGL_MAX_BUFFER_BINDINGS>           uniformBufferObjects;

            //!
            template<GLenum BIND_POINT>
            class BoundBuffer
            {
                    const COpenGLBuffer* boundBuffer;
                    uint64_t lastValidatedBuffer;
                public:
                    BoundBuffer() : lastValidatedBuffer(0)
                    {
                        boundBuffer = NULL;
                    }

                    ~BoundBuffer()
                    {
                        set(NULL);
                    }

                    void set(const COpenGLBuffer* buff);
            };

            //! Indirect
            BoundBuffer<GL_DRAW_INDIRECT_BUFFER> indirectDraw;


            /** We will operate on some assumptions here:

            1) On all GPU's known to me  GPUs MAX_VERTEX_ATTRIB_BINDINGS <= MAX_VERTEX_ATTRIBS,
            so it makes absolutely no sense to support buffer binding mix'n'match as it wouldn't
            get us anything (however if MVAB>MVA then we could have more inputs into a vertex shader).
            Also the VAO Attrib Binding is a VAO state so more VAOs would have to be created in the cache.

            2) Relative byte offset on VAO Attribute spec is capped to 2047 across all GPUs, which makes it
            useful only for specifying the offset from a single interleaved buffer, since we have to specify
            absolute (unbounded) offset and stride when binding a buffer to a VAO bind-point, it makes absolutely
            no sense to use this feature as its redundant.

            So the only things worth tracking for the VAO are:
            1) Element Buffer Binding
            2) Per Attribute (x16)
                A) Enabled (1 bit)
                B) Format (5 bits)
                C) Component Count (3 bits)
                D) Divisors (32bits - no limit)

            Total 16*4+16+16/8+4 = 11 uint64_t

            If we limit divisors artificially to 1 bit

            16/8+16/8+16+4 = 3 uint64_t
            **/
            class COpenGLVAO
            {
                    size_t                      attrOffset[scene::EVAI_COUNT];
                    uint32_t                    attrStride[scene::EVAI_COUNT];
                    //vertices
                    const COpenGLBuffer*        mappedAttrBuf[scene::EVAI_COUNT];
                    //indices
                    const COpenGLBuffer*        mappedIndexBuf;

                    GLuint                      vao;
                    uint64_t                    lastValidated;
                #ifdef _DEBUG
                    COpenGLVAOSpec::HashAttribs debugHash;
                #endif // _DEBUG
                public:
                    _IRR_NO_DEFAULT_FINAL(COpenGLVAO);
                    _IRR_NO_COPY_FINAL(COpenGLVAO);

                    COpenGLVAO(const COpenGLVAOSpec* spec);
                    inline COpenGLVAO(COpenGLVAO&& other)
                    {
                        memcpy(this,&other,sizeof(COpenGLVAO));
                        memset(other.attrOffset,0,sizeof(mappedAttrBuf));
                        memset(other.attrStride,0,sizeof(mappedAttrBuf));
                        memset(other.mappedAttrBuf,0,sizeof(mappedAttrBuf));
                        other.mappedIndexBuf = NULL;
                        other.vao = 0;
                        other.lastValidated = 0;
                    }
                    ~COpenGLVAO();

                    inline const GLuint& getOpenGLName() const {return vao;}


                    inline COpenGLVAO& operator=(COpenGLVAO&& other)
                    {
                        memcpy(this,&other,sizeof(COpenGLVAO));
                        memset(other.mappedAttrBuf,0,sizeof(mappedAttrBuf));
                        memset(other.attrStride,0,sizeof(mappedAttrBuf));
                        memset(other.mappedAttrBuf,0,sizeof(mappedAttrBuf));
                        other.mappedIndexBuf = NULL;
                        other.vao = 0;
                        other.lastValidated = 0;
                        return *this;
                    }


                    void bindBuffers(   const COpenGLBuffer* indexBuf,
                                        const COpenGLBuffer* const* attribBufs,
                                        const size_t offsets[scene::EVAI_COUNT],
                                        const size_t strides[scene::EVAI_COUNT]);

                    inline const uint64_t& getLastBoundStamp() const {return lastValidated;}

                #ifdef _DEBUG
                    inline const COpenGLVAOSpec::HashAttribs& getDebugHash() const {return debugHash;}
                #endif // _DEBUG
            };

            //!
            typedef std::pair<COpenGLVAOSpec::HashAttribs,COpenGLVAO*> HashVAOPair;
            HashVAOPair                 CurrentVAO;
            core::vector<HashVAOPair>    VAOMap;

            inline size_t getVAOCacheSize() const
            {
                return VAOMap.size();
            }

            inline void freeUpVAOCache(bool exitOnFirstDelete)
            {
                for(auto it = VAOMap.begin(); VAOMap.size()>maxVAOCacheSize&&it!=VAOMap.end();)
                {
                    if (it->first==CurrentVAO.first)
                        continue;

                    if (CNullDriver::ReallocationCounter-it->second->getLastBoundStamp()>1000) //maybe make this configurable
                    {
                        delete it->second;
                        it = VAOMap.erase(it);
                        if (exitOnFirstDelete)
                            return;
                    }
                    else
                        it++;
                }
            }

            //! Textures and Samplers
            class STextureStageCache
            {
                const IVirtualTexture* CurrentTexture[MATERIAL_MAX_TEXTURES];
            public:
                STextureStageCache()
                {
                    for (uint32_t i=0; i<MATERIAL_MAX_TEXTURES; ++i)
                    {
                        CurrentTexture[i] = 0;
                    }
                }

                ~STextureStageCache()
                {
                    clear();
                }

                void set(uint32_t stage, const IVirtualTexture* tex)
                {
                    if (stage<MATERIAL_MAX_TEXTURES)
                    {
                        const IVirtualTexture* oldTexture=CurrentTexture[stage];
                        if (tex)
                            tex->grab();
                        CurrentTexture[stage]=tex;
                        if (oldTexture)
                            oldTexture->drop();
                    }
                }

                const IVirtualTexture* operator[](int stage) const
                {
                    if ((uint32_t)stage<MATERIAL_MAX_TEXTURES)
                        return CurrentTexture[stage];
                    else
                        return 0;
                }

                void remove(const IVirtualTexture* tex);

                void clear();
            };

            //!
            STextureStageCache                  CurrentTexture;

            //! Samplers
            uint64_t                            CurrentSamplerHash[MATERIAL_MAX_TEXTURES];
            core::unordered_map<uint64_t,GLuint> SamplerMap;
        };


		//!
		virtual uint32_t getRequiredUBOAlignment() const {return COpenGLExtensionHandler::reqUBOAlignment;}

		//!
		virtual uint32_t getRequiredSSBOAlignment() const {return COpenGLExtensionHandler::reqSSBOAlignment;}

		//!
		virtual uint32_t getRequiredTBOAlignment() const {return COpenGLExtensionHandler::reqTBOAlignment;}

        //!
        virtual uint32_t getMaxComputeWorkGroupSize(uint32_t _dimension) const { return COpenGLExtensionHandler::MaxComputeWGSize[_dimension]; }

    private:
        SAuxContext* getThreadContext_helper(const bool& alreadyLockedMutex, const std::thread::id& tid = std::this_thread::get_id());

        void cleanUpContextBeforeDelete();


        void bindTransformFeedback(ITransformFeedback* xformFeedback, SAuxContext* toContext);



		//! enumeration for rendering modes such as 2d and 3d for minizing the switching of renderStates.
		enum E_RENDER_MODE
		{
			ERM_NONE = 0,	// no render state has been set yet.
			ERM_2D,		// 2d drawing rendermode
			ERM_3D		// 3d rendering mode
		};

		E_RENDER_MODE CurrentRenderMode;
		//! bool to make all renderstates reset if set to true.
		bool ResetRenderStates;

		SGPUMaterial Material, LastMaterial;



		//! inits the parts of the open gl driver used on all platforms
		bool genericDriverInit();
		//! returns a device dependent texture from a software surface (IImage)
		virtual video::ITexture* createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels, const io::path& name, ECOLOR_FORMAT format = ECF_A8R8G8B8);

		// returns the current size of the screen or rendertarget
		virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize() const;

		void createMaterialRenderers();

		core::stringw Name;

		std::string VendorName;

		//! Color buffer format
		ECOLOR_FORMAT ColorFormat; //FIXME

		SIrrlichtCreationParameters Params;

		#ifdef _IRR_WINDOWS_API_
			HDC HDc; // Private GDI Device Context
			HWND Window;
		#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
			CIrrDeviceWin32 *Win32Device;
		#endif
		#endif
		#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
			GLXDrawable Drawable;
			Display* X11Display;
			CIrrDeviceLinux *X11Device;
		#endif
		#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
			CIrrDeviceMacOSX *OSXDevice;
		#endif
		#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
			CIrrDeviceSDL *SDLDevice;
		#endif

        size_t maxALUShaderInvocations;
        size_t maxConcurrentShaderInvocations;
        uint32_t maxShaderComputeUnits;
#ifdef _IRR_COMPILE_WITH_OPENCL_
        cl_device_id clDevice;
        size_t clPlatformIx, clDeviceIx;
#endif // _IRR_COMPILE_WITH_OPENCL_

        FW_Mutex* glContextMutex;
		SAuxContext* AuxContexts;

		E_DEVICE_TYPE DeviceType;
	};

} // end namespace video
} // end namespace irr


#endif // _IRR_COMPILE_WITH_OPENGL_
#endif

