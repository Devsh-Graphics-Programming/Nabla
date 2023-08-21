# How I generated the profiles
```
$ pip install jsonschema
```

```
$ cd ${VULKAN_SDK}/share/vulkan/registry
$ python gen_profiles_file.py --registry "${VULKAN_HEADERS}/registry/vk.xml" --input "PROFILES_DIR" -o "PROFILES_DIR/../profiles.json" --profile-desc "DESCRIPTION" --profile-label "NAME"
```

Note that this python script won't recurse, only profiles directly inside the directory will be intersected.

Profiles resulting from profiles can be further intersected.

## Working around bugs

You need to manually erase any fields dealing with Descriptor Buffer because of: https://github.com/SaschaWillems/VulkanCapsViewer/issues/188