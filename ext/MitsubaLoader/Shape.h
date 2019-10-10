
	void flipNormals(asset::IAssetManager* _assetManager)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager->getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

protected:
	std::string type;
	core::smart_refctd_ptr<asset::ICPUMesh> mesh;
	core::matrix4SIMD transform;

};


}
}
}

#endif