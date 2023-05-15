// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_SKELETON_H_INCLUDED_
#define _NBL_ASSET_I_SKELETON_H_INCLUDED_

#include "nbl/asset/IBuffer.h"
#include "nbl/macros.h"

#include "nbl/core/declarations.h"

namespace nbl::asset
{

//! Class which holds the armature/skeleton.
/** An ISkeleton is nothing more than a Structure of Arrays style collection of
* joints. The two attributes are parent joint IDs and Bind Pose matrices.
*/
template <class BufferType>
class ISkeleton : public virtual core::IReferenceCounted
{
	public:
		using joint_id_t = uint32_t;
		static inline constexpr joint_id_t invalid_joint_id = 0xdeadbeefu;


		inline const auto& getJointNameToIDMap() const
		{
			return m_nameToJointID;
		}
		inline joint_id_t getJointIDFromName(const char* jointName) const
		{
			auto found = m_nameToJointID.find(jointName);
			if (found != m_nameToJointID.end())
				return found->second;
			return invalid_joint_id;
		}

		inline joint_id_t getJointCount() const
		{
			return m_jointCount;
		}

		inline const SBufferBinding<const BufferType>& getParentJointIDBinding() const
		{
			return reinterpret_cast<const SBufferBinding<const BufferType>*>(m_parentJointIDs);
		}

		inline const SBufferBinding<const BufferType>& getDefaultTransforms() const
		{
			return reinterpret_cast<const SBufferBinding<const BufferType>*>(m_defaultTransforms);
		}


	protected:
		ISkeleton(SBufferBinding<BufferType>&& _parentJointIDsBinding, SBufferBinding<BufferType>&& _defaultTransforms, const joint_id_t _jointCount = 0u)
			:	m_nameToJointID(), m_stringPoolSize(0ull), m_stringPool(nullptr), m_jointCount(_jointCount),
				m_parentJointIDs(std::move(_parentJointIDsBinding)), m_defaultTransforms(std::move(_defaultTransforms))
		{
			if (m_jointCount==0u)
				return;

			assert(m_parentJointIDs.buffer->getSize()>=m_parentJointIDs.offset+sizeof(joint_id_t)*m_jointCount);
			assert(m_defaultTransforms.buffer->getSize()>=m_defaultTransforms.offset+sizeof(core::matrix3x4SIMD)*m_jointCount);
		}
		virtual ~ISkeleton()
		{
			clearNames();
		}

		// map must contain one `const char*` per bone
		template<class Comparator>
		inline void setJointNames(const core::map<const char*,joint_id_t,Comparator>& nameToJointIDMap)
	{
			clearNames();

			// size the pool
			for (const auto& mapping : nameToJointIDMap)
				reserveName(mapping.first);
				
			// useless names
			if (m_stringPoolSize==0ull)
				return;

			m_stringPool = _NBL_NEW_ARRAY(char,m_stringPoolSize);
				
			char* outName = m_stringPool;
			for (const auto& mapping : nameToJointIDMap)
				outName = insertName(outName,mapping.first,mapping.second);
		}

		// iterator range must contain one `const char*` per bone
		template<typename NameIterator>
		inline void setJointNames(NameIterator begin, NameIterator end)
		{
			clearNames();

			// size the pool
			for (auto it=begin; it!=end; it++)
				reserveName(*it);
				
			// useless names
			if (m_stringPoolSize==0ull)
				return;

			m_stringPool = _NBL_NEW_ARRAY(char,m_stringPoolSize);
				
			char* outName = m_stringPool;
			joint_id_t jointID = 0u;
			for (auto it=begin; it!=end; it++,jointID++)
				outName = insertName(outName,*it,jointID);
		}

		struct StringComparator
		{
			inline bool operator()(const char* lhs, const char* rhs) const
			{
				return strcmp(lhs,rhs)<0;
			}
		};
		core::map<const char*,joint_id_t,StringComparator> m_nameToJointID;
		size_t m_stringPoolSize;
		char* m_stringPool;

		SBufferBinding<BufferType> m_parentJointIDs,m_defaultTransforms;
		joint_id_t m_jointCount;

	private:
		inline void reserveName(const char* inName)
		{
			const auto nameLen = strlen(inName);
			if (nameLen)
				m_stringPoolSize += nameLen+1ull;
		}
		inline char* insertName(char* outName, const char* inName, joint_id_t jointID)
		{
			const char* name = outName;
			while (*inName) {*(outName++) = *(inName++);}
			if (outName!=name)
			{
				*(outName++) = 0;
				m_nameToJointID.emplace(name,jointID);
			}
			return outName;
		}
		inline void clearNames()
		{
			if (m_stringPool)
				_NBL_DELETE_ARRAY(m_stringPool,m_stringPoolSize);
			m_stringPoolSize = 0ull;
			m_nameToJointID.clear();
		}
};

} // end namespace nbl::asset

#endif

