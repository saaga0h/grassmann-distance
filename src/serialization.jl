# ── JSON3 struct type mappings ────────────────────────────────────────────────

StructTypes.StructType(::Type{EntityInput}) = StructTypes.Struct()
StructTypes.StructType(::Type{GraphConfigInput}) = StructTypes.Struct()
StructTypes.StructType(::Type{BuildParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{QuerySpec}) = StructTypes.Struct()
StructTypes.StructType(::Type{QueryParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{JobParams}) = StructTypes.Struct()

StructTypes.StructType(::Type{PathOutput}) = StructTypes.Struct()
StructTypes.StructType(::Type{ReachableEntry}) = StructTypes.Struct()
StructTypes.StructType(::Type{CommunityOutput}) = StructTypes.Struct()
StructTypes.StructType(::Type{BasinOutput}) = StructTypes.Struct()
StructTypes.StructType(::Type{BridgeOutput}) = StructTypes.Struct()
StructTypes.StructType(::Type{HubEntry}) = StructTypes.Struct()
StructTypes.StructType(::Type{TopologyOutput}) = StructTypes.Struct()
StructTypes.StructType(::Type{BuildOutput}) = StructTypes.Struct()
StructTypes.StructType(::Type{QueryOutput}) = StructTypes.Struct()
StructTypes.StructType(::Type{JobResult}) = StructTypes.Struct()

StructTypes.omitempties(::Type{JobResult}) = (:error, :result)
StructTypes.omitempties(::Type{JobParams}) = (:build, :query)
StructTypes.omitempties(::Type{QuerySpec}) = (:from, :to, :depth)
StructTypes.omitempties(::Type{QueryOutput}) = (:path, :reachable, :topology)

# ── Input parsing ────────────────────────────────────────────────────────────

function parse_job(payload::AbstractVector{UInt8})::JobParams
    JSON3.read(String(payload), JobParams)
end

function parse_job(payload::AbstractString)::JobParams
    JSON3.read(payload, JobParams)
end

# ── Output serialization ────────────────────────────────────────────────────

function serialize_result(result::JobResult)::Vector{UInt8}
    Vector{UInt8}(JSON3.write(result))
end

# ── Graph binary serialization ───────────────────────────────────────────────
# Opaque blob for the client — Julia writes it, Julia reads it.
# Travels through FORGE as base64-encoded string in JSON payloads.

function serialize_graph(graph::GrassmannGraph)::String
    io = IOBuffer()
    Serialization.serialize(io, graph)
    return Base64.base64encode(take!(io))
end

function deserialize_graph(encoded::AbstractString)::GrassmannGraph
    bytes = Base64.base64decode(encoded)
    io = IOBuffer(bytes)
    return Serialization.deserialize(io)::GrassmannGraph
end

# ── Config conversion ───────────────────────────────────────────────────────

function _to_grassmann_config(input::GraphConfigInput)
    dist = input.distance == "chordal" ? :chordal : :geodesic
    return GrassmannConfig(input.k, input.p, dist), GraphConfig(k_graph=input.k_graph, max_chunks=input.max_chunks)
end

# ── Entity input → build_graph arguments ─────────────────────────────────────

function _prepare_build_inputs(entities::Vector{EntityInput})
    entity_ids = [e.id for e in entities]
    chunk_entity_map = String[]
    all_embeddings = Vector{Float64}[]

    for e in entities
        for emb in e.embeddings
            push!(all_embeddings, emb)
            push!(chunk_entity_map, e.id)
        end
    end

    dim = length(all_embeddings[1])
    n_chunks = length(all_embeddings)
    embeddings = Matrix{Float64}(undef, dim, n_chunks)
    for (j, emb) in enumerate(all_embeddings)
        embeddings[:, j] = emb
    end

    return embeddings, entity_ids, chunk_entity_map
end
