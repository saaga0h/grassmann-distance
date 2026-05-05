#!/usr/bin/env julia
# Generate embeddings for the test_corpus/ documents.
#
# Chunks each markdown file by sections (## headings), embeds each chunk
# via Ollama, and writes test_data/corpus_embeddings.json.
#
# Usage:
#   OLLAMA_HOST=<host>:<port> julia --project=. scripts/generate_corpus_embeddings.jl

using HTTP
using JSON3

# ── Configuration ─────────────────────────────────────────────────────────────

const MODEL = "qwen3-embedding:8b"
const CORPUS_DIR = joinpath(@__DIR__, "..", "test_corpus")
const MIN_CHUNK_CHARS = 200   # merge sections shorter than this into previous
const MAX_CHUNK_CHARS = 3000  # split sections longer than this at paragraph boundaries

# ── Resolve Ollama host ──────────────────────────────────────────────────────

ollama_host = get(ENV, "OLLAMA_HOST", nothing)
if ollama_host === nothing
    error("OLLAMA_HOST environment variable is required (e.g. OLLAMA_HOST=<host>:<port>)")
end

embed_url = "http://$(ollama_host)/api/embed"

# ── Chunking ─────────────────────────────────────────────────────────────────

function chunk_markdown(text::String; min_chars=MIN_CHUNK_CHARS, max_chars=MAX_CHUNK_CHARS)
    chunks = String[]
    current = ""
    current_heading = ""

    for line in eachline(IOBuffer(text))
        # Detect ## or ### heading (not # title)
        if match(r"^#{2,3}\s", line) !== nothing
            # Flush current chunk
            if !isempty(strip(current))
                push!(chunks, strip(current))
            end
            current = line * "\n"
            current_heading = line
        else
            current *= line * "\n"
        end
    end
    # Flush last chunk
    if !isempty(strip(current))
        push!(chunks, strip(current))
    end

    # Merge short chunks into previous
    merged = String[]
    for chunk in chunks
        if !isempty(merged) && length(merged[end]) + length(chunk) < min_chars * 2
            # If current chunk is too short, merge with previous
            if length(chunk) < min_chars
                merged[end] = merged[end] * "\n\n" * chunk
                continue
            end
        end
        push!(merged, chunk)
    end

    # Split overly long chunks at paragraph boundaries
    result = String[]
    for chunk in merged
        if length(chunk) <= max_chars
            push!(result, chunk)
        else
            # Split at double newlines
            paragraphs = split(chunk, r"\n\n+")
            current_split = ""
            for para in paragraphs
                if length(current_split) + length(para) > max_chars && !isempty(strip(current_split))
                    push!(result, strip(current_split))
                    current_split = para
                else
                    current_split *= (isempty(current_split) ? "" : "\n\n") * para
                end
            end
            if !isempty(strip(current_split))
                push!(result, strip(current_split))
            end
        end
    end

    return result
end

# ── Discover and chunk documents ─────────────────────────────────────────────

files = sort(filter(f -> endswith(f, ".md"), readdir(CORPUS_DIR)))

all_chunks = NamedTuple{(:doc, :chunk_idx, :text), Tuple{String, Int, String}}[]

for file in files
    doc_name = replace(file, ".md" => "")
    content = read(joinpath(CORPUS_DIR, file), String)
    chunks = chunk_markdown(content)
    for (i, chunk) in enumerate(chunks)
        push!(all_chunks, (doc=doc_name, chunk_idx=i, text=chunk))
    end
end

println("=== Corpus Embedding Generation ===")
println("Model:     $MODEL")
println("Endpoint:  $embed_url")
println("Documents: $(length(files))")
println("Chunks:    $(length(all_chunks))")
println()

# Show per-document chunk counts
for file in files
    doc_name = replace(file, ".md" => "")
    n = count(c -> c.doc == doc_name, all_chunks)
    println("  $(rpad(doc_name, 50)) $n chunks")
end
println()

# ── Generate embeddings ──────────────────────────────────────────────────────

entries = []
dimension = nothing

for (i, chunk) in enumerate(all_chunks)
    print("  [$i/$(length(all_chunks))] $(chunk.doc) #$(chunk.chunk_idx) ... ")

    body = Dict("model" => MODEL, "input" => chunk.text)
    resp = HTTP.post(embed_url;
        headers = ["Content-Type" => "application/json"],
        body = JSON3.write(body),
        retry = false,
    )

    data = JSON3.read(resp.body)
    embedding = collect(Float64, data.embeddings[1])

    if dimension === nothing
        global dimension = length(embedding)
        println("dim=$dimension")
    else
        @assert length(embedding) == dimension
        println("ok")
    end

    push!(entries, Dict(
        "id"        => "$(chunk.doc)_$(lpad(chunk.chunk_idx, 2, '0'))",
        "doc"       => chunk.doc,
        "chunk_idx" => chunk.chunk_idx,
        "text"      => chunk.text,
        "embedding" => embedding,
    ))
end

# ── Write output ─────────────────────────────────────────────────────────────

output_path = joinpath(@__DIR__, "..", "test_data", "corpus_embeddings.json")
mkpath(dirname(output_path))

result = Dict(
    "model"     => MODEL,
    "dimension" => dimension,
    "n_docs"    => length(files),
    "n_chunks"  => length(all_chunks),
    "entries"   => entries,
)

open(output_path, "w") do io
    JSON3.pretty(io, result)
end

println()
println("Written $(length(entries)) chunk embeddings (dim=$dimension) to $output_path")
