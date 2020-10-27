// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_VIDEO_NULL_H_INCLUDED__
#define __C_VIDEO_NULL_H_INCLUDED__

#include "irrlicht.h"

#include "IVideoDriver.h"
#include "IFileSystem.h"
#include "irr/asset/IMesh.h"
#include "irr/video/IGPUMeshBuffer.h"
#include "CFPSCounter.h"
#include "SExposedVideoData.h"
#include "FW_Mutex.h"


#ifdef _MSC_VER
#pragma warning( disable: 4996)
#endif

namespace irr
{
namespace video
{

class CNullDriver : public IVideoDriver
{
    protected:
		//! destructor
		virtual ~CNullDriver();

		//! inits the parts of the driver used on all platforms
		virtual bool genericDriverInit(asset::IAssetManager* assMgr);

	public:
        static FW_AtomicCounter ReallocationCounter;
        static int32_t incrementAndFetchReallocCounter();

		//! constructor
		CNullDriver(IrrlichtDevice* dev, io::IFileSystem* io, const SIrrlichtCreationParameters& _params);

        bool bindGraphicsPipeline(const video::IGPURenderpassIndependentPipeline* _gpipeline) override { return false; }

        bool bindComputePipeline(const video::IGPUComputePipeline* _cpipeline) override { return false; }

        bool bindDescriptorSets(E_PIPELINE_BIND_POINT _pipelineType, const IGPUPipelineLayout* _layout,
            uint32_t _first, uint32_t _count, const IGPUDescriptorSet* const* _descSets, core::smart_refctd_dynamic_array<uint32_t>* _dynamicOffsets) override 
        { 
            return false; 
        }

        bool dispatch(uint32_t _groupCountX, uint32_t _groupCountY, uint32_t _groupCountZ) override { return false; }
        bool dispatchIndirect(const IGPUBuffer* _indirectBuf, size_t _offset) override { return false; }

        bool pushConstants(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values) override
        {
            if (!_layout || !_values)
                return false;
            if (!_size)
                return false;
            if (!_stages)
                return false;
            if (!core::is_aligned_to(_offset, 4u))
                return false;
            if (!core::is_aligned_to(_size, 4u))
                return false;
            if (_offset >= IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
                return false;
            if ((_offset+_size) > IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
                return false;

            asset::SPushConstantRange updateRange;
            updateRange.offset = _offset;
            updateRange.size = _size;

#ifdef _IRR_DEBUG
            //TODO validation:
            /*
            For each byte in the range specified by offset and size and for each shader stage in stageFlags,
            there must be a push constant range in layout that includes that byte and that stage
            */
            for (const auto& rng : _layout->getPushConstantRanges())
            {
                /*
                For each byte in the range specified by offset and size and for each push constant range that overlaps that byte,
                stageFlags must include all stages in that push constant range’s VkPushConstantRange::stageFlags
                */
                if (updateRange.overlap(rng) && ((_stages & rng.stageFlags) != rng.stageFlags))
                    return false;
            }
#endif//_IRR_DEBUG

            return true;
        }

		//!
        virtual bool initAuxContext() {return false;}
        virtual bool deinitAuxContext() {return false;}

		virtual bool beginScene(bool backBuffer=true, bool zBuffer=true,
				SColor color=SColor(255,0,0,0),
				const SExposedVideoData& videoData=SExposedVideoData(),
				core::rect<int32_t>* sourceRect=0) override;

		virtual bool endScene() override;

		//!
		virtual void issueGPUTextureBarrier() override {}

        //! GPU fence, is signalled when preceeding GPU work is completed
        virtual core::smart_refctd_ptr<IDriverFence> placeFence(const bool& implicitFlushWaitSameThread=false) override {return nullptr;}

		//!
		virtual CPropertyPoolHandler* getDefaultPropertyPoolHandler() const override { return m_propertyPoolHandler.get(); }

		//! gets the area of the current viewport
		virtual const core::rect<int32_t>& getViewPort() const override;

        virtual void drawMeshBuffer(const video::IGPUMeshBuffer* mb) override;

		//! Indirect Draw
		virtual void drawArraysIndirect(const asset::SBufferBinding<IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) override
        {
        }
		virtual void drawIndexedIndirect(const asset::SBufferBinding<IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        asset::E_INDEX_TYPE indexType, const IGPUBuffer* indexBuff,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) override
        {
        }

		//! get color format of the current color buffer
		virtual asset::E_FORMAT getColorFormat() const;

		//! get screen size
		virtual const core::dimension2d<uint32_t>& getScreenSize() const;

		//! get render target size
		virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize() const;

		// get current frames per second value
		virtual int32_t getFPS() const;

		//! returns amount of primitives (mostly triangles) were drawn in the last frame.
		//! very useful method for statistics.
		virtual uint32_t getPrimitiveCountDrawn( uint32_t param = 0 ) const;

		//! \return Returns the name of the video driver. Example: In case of the DIRECT3D8
		//! driver, it would return "Direct3D8.1".
		virtual const wchar_t* getName() const;

		//!
		virtual void beginQuery(IQueryObject* query);
		virtual void endQuery(IQueryObject* query);

		//! Only used by the engine internally.
		/** Used to notify the driver that the window was resized. */
		virtual void OnResize(const core::dimension2d<uint32_t>& size);

		//! Returns driver and operating system specific data about the IVideoDriver.
		virtual const SExposedVideoData& getExposedVideoData();

		//! Returns type of video driver
		virtual E_DRIVER_TYPE getDriverType() const override;

		//! Returns the graphics card vendor name.
		virtual std::string getVendorInfo() {return "Not available on this driver.";}

		//! Returns the maximum texture size supported.
		virtual const uint32_t* getMaxTextureSize(IGPUImageView::E_TYPE type) const override;

		//!
		const CDerivativeMapCreator* getDerivativeMapCreator() const override { return DerivativeMapCreator.get(); };

	protected:
        void bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count,
										const IGPUDescriptorSet* const* _descSets, const IGPUPipelineLayout** _destPplnLayouts);

		SIrrlichtCreationParameters Params;

        IQueryObject* currentQuery[EQOT_COUNT];

		io::IFileSystem* FileSystem;

		core::rect<int32_t> ViewPort;

		CFPSCounter FPSCounter;

		uint32_t PrimitivesDrawn;

		core::smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
		core::smart_refctd_ptr<CDerivativeMapCreator> DerivativeMapCreator;

		SExposedVideoData ExposedData;

		uint32_t MaxTextureSizes[IGPUImageView::ET_COUNT][3];
	};

	IVideoDriver* createNullDriver(IrrlichtDevice* device, io::IFileSystem* io, const SIrrlichtCreationParameters& screenSize);

} // end namespace video
} // end namespace irr


#endif
