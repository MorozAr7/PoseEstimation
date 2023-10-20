IMG_W = 640
IMG_H = 480

# Create offscreen window so that no window will be opened
base = ShowBase(fStartDirect=True, windowType='offscreen')

# Create frame buffer properties
fb_prop = FrameBufferProperties()
fb_prop.setRgbColor(True)
# Only render RGB with 8 bit for each channel, no alpha channel
fb_prop.setRgbaBits(8, 8, 8, 0)
fb_prop.setDepthBits(24)

# Create window properties
win_prop = WindowProperties.size(IMG_W, IMG_H)

# Create window (offscreen)
window = base.graphicsEngine.makeOutput(base.pipe, "cameraview", 0, fb_prop, win_prop, GraphicsPipe.BFRefuseWindow)

# Create display region
# This is the actual region used where the image will be rendered
disp_region = window.makeDisplayRegion()


# Assign a camera for the region
disp_region.setCamera(cam_obj)

bgr_tex = Texture()
window.addRenderTexture(bgr_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)



# Now we can render the frame manually
base.graphicsEngine.renderFrame()

# Get the frame data as numpy array
bgr_img = np.frombuffer(bgr_tex.getRamImage(), dtype=np.uint8)
bgr_img.shape = (bgr_tex.getYSize(), bgr_tex.getXSize(), bgr_tex.getNumComponents())
