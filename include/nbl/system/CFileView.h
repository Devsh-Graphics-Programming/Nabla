#ifndef _NBL_SYSTEM_C_FILE_VIEW_H_
#define _NBL_SYSTEM_C_FILE_VIEW_H_


#include "nbl/system/IFile.h"


namespace nbl::system
{

class IFileView : public IFile
{
	public:
		inline size_t getSize() const override final
		{
			return m_size;
		}

	protected:
		inline IFileView(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, const time_point_t _initialModified, void* buffer, size_t fileSize) :
			IFile(std::move(_name),_flags,_initialModified), m_buffer(buffer), m_size(fileSize) {}

		inline const void* getMappedPointer_impl() const override final
		{
			return m_buffer;
		}
		inline void* getMappedPointer_impl() override final
		{
			return m_buffer;
		}

		//
		void* const m_buffer;
		const size_t m_size;
};

template<typename allocator_t>
class CFileView : public IFileView
{
		// what do I even need this "friend" for?
		friend class CAPKResourcesArchive;

	public:
		// constructor for making a file with memory already allocated by the allocator
		inline CFileView(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, const time_point_t _initialModified, void* buffer, const size_t fileSize, allocator_t&& _allocator={}) :
			IFileView(std::move(_name),_flags,_initialModified,buffer,fileSize), allocator(std::move(_allocator)) {}

		// 
		static inline core::smart_refctd_ptr<CFileView<allocator_t>> create(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, const time_point_t _initialModified, size_t fileSize, allocator_t&& _allocator={})
		{
			auto mem = reintepret_cast<std::byte*>(_allocator.alloc(fileSize));
			if (!mem)
				return nullptr;
			auto retval = new CFileView(std::move(_name),_flags,_initialModified,mem,fileSize,std::move(_allocator));
			return core::smart_refctd_ptr(retval,core::dont_grab);
		}

		// 
		static inline core::smart_refctd_ptr<CFileView<allocator_t>> create(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, size_t fileSize, allocator_t&& _allocator={})
		{
			return create(std::move(_name),_flags,std::chrono::utc_clock::now(),fileSize,_allocator);
		}

	protected:
		inline ~CFileView()
		{
			if (m_buffer)
				allocator.dealloc(m_buffer, m_size);
		}

		allocator_t allocator;
};


// Forward declare the null file allocator callback
class CNullAllocator;

//
template<>
class CFileView<CNullAllocator> : public IFileView
{
	public:
		CFileView(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, const time_point_t _initialModified, void* buffer, const size_t fileSize) :
			IFileView(std::move(_name),_flags,_initialModified,buffer,fileSize) {}
		// the `_allocator` parameter is useless and unused but its necessary to have a uniform constructor signature across all specializations (saves us headaches in `CFileArchive`)
		CFileView(path&& _name, const core::bitflag<E_CREATE_FLAGS> _flags, const time_point_t _initialModified, void* buffer, const size_t fileSize, CNullAllocator&& _allocator) :
			CFileView(std::move(_name),_flags,_initialModified,buffer,fileSize) {}

	protected:
		~CFileView() = default;
};

}

#endif