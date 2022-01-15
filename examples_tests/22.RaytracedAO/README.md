# Using the Renderer

## How the Renderer works

* It starts by rendering the scene with each sensor and will stop when enough samples is taken.
	* To skip this part, press the `END` Key. (more detail below)
---
* Then you'll have full control of the camera, you can take snapshots, move around and have fun :)
	* To skip this part pass `-TERMINATE` as a cmd argument when executing the renderer (more detail below).
---
* Before Exiting from the Renderer, the very last view will be rendered and denoised to files named like `LastView_spaceship_Sensor_0`

## CommandLine Help
```
Parameters:
-SCENE=sceneMitsubaXMLPathOrZipAndXML
-TERMINATE

Description and usage: 

-SCENE:
	some/path extra/path which will make it skip the file choose dialog

	NOTE: If the scene path contains space, put it between quotation marks

-TERMINATE:
	It will make the app stop when the required amount of samples has been renderered (its in the Mitsuba Scene metadata) and obviously take screenshot when quitting
	

Example Usages :
	raytracedao.exe -SCENE=../../media/kitchen.zip scene.xml -TERMINATE
	raytracedao.exe -SCENE="../../media/my good kitchen.zip" scene.xml -TERMINATE
	raytracedao.exe -SCENE="../../media/my good kitchen.zip scene.xml" -TERMINATE
	raytracedao.exe -SCENE="../../media/extraced folder/scene.xml" -TERMINATE
```


## New mitsuba properties and tags 
Multiple Sensor tags in mitsuba XML's is now supported. This feature helps you have multiple views with different camera and film parameters without needing to execute the renderer and load again.

You can switch between those sensors using `PAGE UP/DOWN` Keys defined in more detail below.

### Properties added to \<sensor\>:

| Property Name | Description           | Type  | Default Value                            |
|---------------|-----------------------|-------|------------------------------------------|
|   moveSpeed   | Camera Movement Speed | float | NaN -> Will be deduced from scene bounds |
|   zoomSpeed   | Camera Zoom Speed     | float | NaN -> Will be deduced from scene bounds |
|  rotateSpeed  | Camera Rotation Speed | float | 300.0 |

### Properties added to \<film\>
| Property Name  | Description                                                                            | Type   | Default Value                                                                                                                                                            |
|----------------|----------------------------------------------------------------------------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| outputFilePath | Final Render Output Path;<br>Denoised Render will have "_denoised" suffix added to it. | string | Render_{SceneName}_Sensor_{SensorIdx}.exr<br>{SceneName} is the filename of the xml or zip loaded.<br>{SensorIdx} is the index of the Sensor in xml used for the render. |
|   bloomScale   | Denoiser Bloom Scale                                                                   | float  | 0.1                                                                                                                                                                      |
| bloomIntensity | Denoiser Bloom Intensity                                                               | float  | 0.1                                                                                                                                                                      |
|  bloomFilePath | Lens Flare File Path                                                                   | string | "../../media/kernels/physical_flare_512.exr"                                                                                                                             |
|   tonemapper   | Tonemapper Settings for Denoiser                                                       | string | "ACES=0.4,0.8"                                                                                                                                                           |
|   highQualityEdges   | Number in pixels (prevously was bool) to add to borders for more accurate denoising            | integer| 0                                                                                                                                                           |

### Example of a sensor using all new properties described above.
```xml
<sensor type="perspective" >
	<float name="fov" value="60" />
	<float name="moveSpeed" value="100.0" />
	<float name="zoomSpeed" value="1.0" />
	<float name="rotateSpeed" value="300.0" />
	<transform name="toWorld" >
		<matrix value="-0.89874 -0.0182716 -0.4381 1.211 0 0.999131 -0.0416703 1.80475 0.438481 -0.0374507 -0.89796 3.85239 0 0 0 1"/>
	</transform>
	<sampler type="sobol" >
		<integer name="sampleCount" value="1024" />
	</sampler>
	<film type="ldrfilm" >
		<string name="outputFilePath" value="C:\Users\MyUser\Desktop\MyRender.exr" />
		<integer name="width" value="1920" />
		<integer name="height" value="1080" />
		<string name="fileFormat" value="png" />
		<string name="pixelFormat" value="rgb" />
		<float name="gamma" value="2.2" />
		<boolean name="banner" value="false" />
		<float name="bloomScale" value="0.1" />
		<float name="bloomIntensity" value="0.1" />
		<string name="bloomFilePath" value="../../media/kernels/physical_flare_512.exr" />
		<string name="tonemapper" value="ACES=0.4,0.8" />
		<rfilter type="tent" />
	</film>
</sensor>
```

## Mouse

| Button              | Description                             |
|---------------------|-------------------------------------------------|
| Left Mouse Button   | Drag to Look around                             |
| Mouse Wheel Scroll  | Zoom In/Out (you can set the speed via mitsuba) |
| Right Mouse Button  | Drag to Move around                             |
| Middle Mouse Button | Drag to Move around                             |

## Keyboard
| Key       | Description                                                                                                            |
|-----------|------------------------------------------------------------------------------------------------------------------------|
| Q         | Press to Quit the Renderer                                                                                             |
| END       | Press to Skip Current Render and "free" the camera                                                                     |
| PAGE_UP   | Press to switch view to the 'next' sensor defined in mitsuba.                                                          |
| PAGE_DOWN | Press to switch view to the 'previous' sensor defined in mitsuba.                                                      |
| HOME      | Press to reset the camera to the initial view. (Usefull when you're lost and you want to go back to where you started) |
| P         | Press to take a snapshot when moving around (will be denoised)                                                         |
| L         | Press to log the current progress percentage and samples rendered.                                                     |
| B         | Toggle between Path Tracing and Albedo preview, allows you to position the camera more responsively in complex scenes. |

## Denoiser Hook
`denoiser_hook.bat` is a script that you can call to denoise your rendered images.

Example:
```
denoiser_hook.bat "Render_scene_Sensor_0.exr" "Render_scene_Sensor_0_albedo.exr" "Render_scene_Sensor_0_normal.exr" "../../media/kernels/physical_flare_512.exr" 0.1 0.3 "ACES=0.4,0.8"
```

Parameters:
1. ColorFile
2. AlbedoFile
3. NormalFile
4. BloomPsfFilePath
5. BloomScale
6. BloomIntensity
7. TonemapperArgs(string)


## Testing in batch

Run `test.bat` to batch render all of the files referenced in `test_scenes.txt`

Here is an example of  `test_scenes.txt`:
```
"../../media/mitsuba/staircase2.zip scene.xml"
"C:\Mitsuba\CoolLivingRoom\scene.xml"
"C:\Mitsuba\CoolKitchen.zip scene.xml"
"../../MitsubaFiles/spaceship.zip scene.xml"
; Here is my Commented line that batch file will skip (started with semicolons)
; "relative/dir/from/bin/folder/to/scene.zip something.xml
```
lines with semicolons will be skipped.