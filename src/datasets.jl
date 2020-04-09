# Make functions to return available content and style images

"""
style_images()

Load packaged style images
"""
style_images() = joinpath.(@__DIR__, "style_images", readdir("style_images"))

"""
content_images()

Load packaged style images.
"""
function content_images()
    content = joinpath.(@__DIR__, "content_images", readdir("content_images"))
    pop!(content)  # Get rid of mould_rosen directory
    append!(content, joinpath.(@__DIR__, "content_images/mould_rosin_npr_general",
                                          readdir("content_images/mould_rosin_npr_general")))
end
