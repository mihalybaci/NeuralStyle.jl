# Make functions to return available content and style images


"""
list_styles()

Gets a list of style images
"""
function list_styles()
    style_dir = joinpath(@__DIR__, "..", "style_images")
    return joinpath.(style_dir, readdir(style_dir))
end
"""
list_contents()

Gets a list of content images
"""
function list_contents()
    content_dir = joinpath.(@__DIR__, "..", "content_images")
    npr_general = joinpath.(@__DIR__, "..", "content_images", "mould_rosin_npr_general")
    content = joinpath.(content_dir, readdir(content_dir))
    pop!(content)  # Get rid of mould_rosen directory
    append!(content, joinpath.(npr_general, readdir(npr_general)))
end
