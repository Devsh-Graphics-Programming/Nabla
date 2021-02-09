// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_SKELETON_H_INCLUDED__
#define __NBL_ASSET_I_SKELETON_H_INCLUDED__

#include "nbl/macros.h"

#include "nbl/core/core.h"

namespace nbl
{
namespace asset
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
			_NBL_STATIC_INLINE_CONSTEXPR joint_id_t invalid_joint_id = 0xffffu;


			inline joint_id_t getJointIDFromName(const char* jointName) const
			{
				auto found = m_nameToJointID.find(jointName);
				if (found != m_nameToJointID.end())
					return found->second;
				return invalid_joint_id;
			}

			inline uint32_t getJointCount() const
			{
				return jointCount;
			}

			inline SBufferBinding<BufferType>& getParentJointIDBinding()
			{
				return m_parentJointIDs;
			}
			inline const SBufferBinding<BufferType>& getParentJointIDBinding() const
			{
				return m_parentJointIDs;
			}

			inline SBufferBinding<BufferType>& getInverseBindPosesBinding()
			{
				return m_inverseBindPoses;
			}
			inline const SBufferBinding<BufferType>& getInverseBindPosesBinding() const
			{
				return m_inverseBindPoses;
			}


		protected:
			ISkeleton(SBufferBinding<BufferType>&& _parentJointIDsBinding, SBufferBinding<BufferType>&& _inverseBindPosesBinding)
				:	m_nameToJointID(), m_stringPoolSize(0ull), m_stringPool(nullptr), m_jointCount(0u),
					m_parentJointIDs(std::move(_parentJointIDsBinding)), m_inverseBindPoses(std::move(_inverseBindPosesBinding))
			{
			}
			virtual ~ISkeleton()
			{
				_NBL_DELETE_ARRAY(m_stringPool,m_stringPoolSize);
			}

			// iterator range must contain one `const char*` per bone
			template<typename NameIterator>
			inline void setJointNames(NameIterator begin, NameIterator end)
			{
				// deinit
				if (m_stringPool)
					_NBL_DELETE_ARRAY(m_stringPool,m_stringPoolSize);
				m_stringPoolSize = 0ull;
				m_nameToJointID.clear();

				// size the pool
				for (auto it=begin; it!=end; it++)
				{
					const char* inName = *it;
					const auto nameLen = strlen(inName);
					if (nameLen)
						m_stringPoolSize += nameLen+1ull;
				}
				
				// useless names
				if (m_stringPoolSize==0ull)
					return;

				m_stringPool = _NBL_NEW_ARRAY(char,m_stringPoolSize);
				
				char* outName = m_stringPool;
				joint_id_t jointID = 0u;
				for (auto it=begin; it!=end; it++,jointID++)
				{
					const char* name = outName;

					const char* inName = *it;
					while (*inName) {*(outName++) = *(inName++);}
					if (outName!=name)
					{
						*(outName++) = '0';
						m_nameToJointID.emplace(name,jointID);
					}
				}
			}

			struct StringComparator
			{
				inline bool operator()(const char* lhs, const char* rhs) const
				{
					return strcmp(lhs,rhs);
				}
			};
			core::map<const char*,joint_id_t,StringComparator> m_nameToJointID;
			size_t m_stringPoolSize;
			char* m_stringPool;

			uint32_t m_jointCount;

			SBufferBinding<BufferType> m_parentJointIDs,m_inverseBindPoses;
	};

} // end namespace asset
} // end namespace nbl

#endif

