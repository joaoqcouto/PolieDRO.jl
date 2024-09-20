using Documenter
using PolieDRO

makedocs(;
    modules=[PolieDRO],
    repo = Documenter.Remotes.GitHub("joaoqcouto", "PolieDRO.jl"),
    doctest=true,
    clean=true,
    checkdocs=:none,
    format=Documenter.HTML(mathengine=Documenter.MathJax2()),
    sitename = "PolieDRO.jl",
    authors="João Couto",
    pages=[
        "Home" => "index.md",
    ],
)