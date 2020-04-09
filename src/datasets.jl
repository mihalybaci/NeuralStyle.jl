# Make functions to return available content and style images

"""
load_styles()

Load packaged style images
"""
load_styles() = joinpath.(@__DIR__, "..", "style_images", readdir("style_images"))

"""
load_contents()

Load packaged style images.
"""
function load_contents()
    content = joinpath.(@__DIR__, "..", "content_images", readdir("content_images"))
    pop!(content)  # Get rid of mould_rosen directory
    append!(content, joinpath.(@__DIR__, "..", "content_images/mould_rosin_npr_general",
                                          readdir("content_images/mould_rosin_npr_general")))
end
