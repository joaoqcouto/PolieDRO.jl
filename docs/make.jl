using Documenter
using PolieDRO

DocMeta.setdocmeta!(PolieDRO, :DocTestSetup, :(using PolieDRO); recursive = true)

makedocs(;
  modules = [PolieDRO],
  checkdocs = :exports,
  doctest = true,
  linkcheck = false,
  authors = "João Couto",
  repo = Documenter.Remotes.GitHub("joaoqcouto", "PolieDRO.jl"),
  sitename = "PolieDRO.jl",
  format = Documenter.HTML(;
    prettyurls = true,
    canonical = "https://joaoqcouto.github.io/PolieDRO.jl",
  ),
  pages = [
    "Introduction" => "index.md",
    "Models" => "models.md",
    "API reference" => "reference.md",
  ],
)

deploydocs(; repo = "github.com/joaoqcouto/PolieDRO.jl", push_preview = false)