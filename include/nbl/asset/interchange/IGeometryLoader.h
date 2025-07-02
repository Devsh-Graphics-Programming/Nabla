// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_LOADER_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_LOADER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"
#include "nbl/asset/utils/CGeometryManipulator.h"


namespace nbl::asset
{

class IGeometryLoader : public IAssetLoader
{
	public:
		virtual inline uint64_t getSupportedAssetTypesBitfield() const override {return IAsset::ET_GEOMETRY;}

	protected:
		inline IGeometryLoader() {}

		template<bool AdoptMemory=false>
		static inline IGeometry<ICPUBuffer>::SDataView createView(
			const E_FORMAT format, const size_t elementCount, const void* data=nullptr,
			core::smart_refctd_ptr<core::refctd_memory_resource>&& memoryResource=nullptr, const size_t alignment=_NBL_SIMD_ALIGNMENT
		)
		{
			const auto stride = getTexelOrBlockBytesize(format);
			core::smart_refctd_ptr<ICPUBuffer> buffer;
			if constexpr (AdoptMemory)
				buffer = ICPUBuffer::create({{stride*elementCount},const_cast<void*>(data),std::move(memoryResource),alignment},core::adopt_memory);
			else
				buffer = ICPUBuffer::create({{stride*elementCount},const_cast<void*>(data),std::move(memoryResource),alignment});
			IGeometry<ICPUBuffer>::SDataView retval = {
				.composed = {
					.stride = stride,
					.format = format,
					.rangeFormat = IGeometryBase::getMatchingAABBFormat(format)
				},
				.src = {.offset=0,.size=buffer->getSize(),.buffer=std::move(buffer)}
			};
			if (data)
			{
				CGeometryManipulator::recomputeContentHash(retval);
				CGeometryManipulator::computeRange(retval);
			}
			return retval;
		}
		// creates a View from a mapped file
		class CFileMemoryResource final : public core::refctd_memory_resource
		{
			public:
				inline CFileMemoryResource(core::smart_refctd_ptr<system::IFile>&& _file) : m_file(std::move(_file)) {}

				inline void* allocate(std::size_t bytes, std::size_t alignment) override
				{
					assert(false); // should never be called
				}
				inline void deallocate(void* p, std::size_t bytes, std::size_t alignment) override
				{
					assert(m_file);
					auto* const basePtr = reinterpret_cast<const uint8_t*>(m_file->getMappedPointer());
					assert(basePtr && basePtr<=p && p<=basePtr+m_file->getSize());
				}

			protected:
				core::smart_refctd_ptr<system::IFile> m_file;
		};
		static inline IGeometry<ICPUBuffer>::SDataView createView(const E_FORMAT format, const size_t elementCount, core::smart_refctd_ptr<system::IFile>&& file, const size_t offsetInFile)
		{
			if (auto* const basePtr=reinterpret_cast<const uint8_t*>(file->getMappedPointer()); basePtr)
			{
				auto resource = core::make_smart_refctd_ptr<CFileMemoryResource>(std::move(file));
				auto* const data = basePtr+offsetInFile;
				return createView<true>(format,elementCount,data,std::move(resource),0x1ull<<hlsl::findLSB(ptrdiff_t(data)));
			}
			else
			{
				auto view = createView(format,elementCount);
				system::IFile::success_t success;
				file->read(success,reinterpret_cast<uint8_t*>(view.src.buffer->getPointer())+view.src.offset,offsetInFile,view.src.actualSize());
				if (success)
				{
					CGeometryManipulator::recomputeContentHash(view);
					CGeometryManipulator::computeRange(view);
					return view;
				}
			}
			return {};
		}

	private:
};

}

#endif
