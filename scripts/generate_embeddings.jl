#!/usr/bin/env julia
# Generate real embedding test data via Ollama.
#
# Calls qwen3-embedding:8b through the Ollama /api/embed endpoint to produce
# 4096D embeddings for a curated word list organised into semantic clusters.
#
# Usage:
#   OLLAMA_HOST=<host>:<port> julia --project=. scripts/generate_embeddings.jl

using HTTP
using JSON3

# ── Configuration ─────────────────────────────────────────────────────────────

const MODEL = "qwen3-embedding:8b"

const DATASET = [
    # ── Citrus (15) ──────────────────────────────────────────────────────────
    ("citrus_01", "orange",      "citrus"),
    ("citrus_02", "mandarin",    "citrus"),
    ("citrus_03", "lemon",       "citrus"),
    ("citrus_04", "lime",        "citrus"),
    ("citrus_05", "grapefruit",  "citrus"),
    ("citrus_06", "tangerine",   "citrus"),
    ("citrus_07", "clementine",  "citrus"),
    ("citrus_08", "kumquat",     "citrus"),
    ("citrus_09", "yuzu",        "citrus"),
    ("citrus_10", "pomelo",      "citrus"),
    ("citrus_11", "bergamot",    "citrus"),
    ("citrus_12", "citron",      "citrus"),
    ("citrus_13", "blood orange","citrus"),
    ("citrus_14", "key lime",    "citrus"),
    ("citrus_15", "satsuma",     "citrus"),

    # ── Tree fruits (15) ─────────────────────────────────────────────────────
    ("tree_01", "apple",       "tree_fruit"),
    ("tree_02", "pear",        "tree_fruit"),
    ("tree_03", "peach",       "tree_fruit"),
    ("tree_04", "plum",        "tree_fruit"),
    ("tree_05", "cherry",      "tree_fruit"),
    ("tree_06", "apricot",     "tree_fruit"),
    ("tree_07", "nectarine",   "tree_fruit"),
    ("tree_08", "quince",      "tree_fruit"),
    ("tree_09", "persimmon",   "tree_fruit"),
    ("tree_10", "fig",         "tree_fruit"),
    ("tree_11", "pomegranate", "tree_fruit"),
    ("tree_12", "mulberry",    "tree_fruit"),
    ("tree_13", "loquat",      "tree_fruit"),
    ("tree_14", "crabapple",   "tree_fruit"),
    ("tree_15", "damson",      "tree_fruit"),

    # ── Tropical (15) ────────────────────────────────────────────────────────
    ("tropical_01", "mango",        "tropical"),
    ("tropical_02", "pineapple",    "tropical"),
    ("tropical_03", "papaya",       "tropical"),
    ("tropical_04", "banana",       "tropical"),
    ("tropical_05", "coconut",      "tropical"),
    ("tropical_06", "guava",        "tropical"),
    ("tropical_07", "passion fruit","tropical"),
    ("tropical_08", "lychee",       "tropical"),
    ("tropical_09", "dragon fruit", "tropical"),
    ("tropical_10", "starfruit",    "tropical"),
    ("tropical_11", "jackfruit",    "tropical"),
    ("tropical_12", "durian",       "tropical"),
    ("tropical_13", "rambutan",     "tropical"),
    ("tropical_14", "mangosteen",   "tropical"),
    ("tropical_15", "plantain",     "tropical"),

    # ── Berries (15) ─────────────────────────────────────────────────────────
    ("berry_01", "strawberry",  "berry"),
    ("berry_02", "blueberry",   "berry"),
    ("berry_03", "raspberry",   "berry"),
    ("berry_04", "blackberry",  "berry"),
    ("berry_05", "cranberry",   "berry"),
    ("berry_06", "gooseberry",  "berry"),
    ("berry_07", "boysenberry", "berry"),
    ("berry_08", "elderberry",  "berry"),
    ("berry_09", "lingonberry", "berry"),
    ("berry_10", "huckleberry", "berry"),
    ("berry_11", "acai berry",  "berry"),
    ("berry_12", "goji berry",  "berry"),
    ("berry_13", "currant",     "berry"),
    ("berry_14", "loganberry",  "berry"),
    ("berry_15", "cloudberry",  "berry"),

    # ── Vegetables (15) ──────────────────────────────────────────────────────
    ("veg_01", "tomato",    "vegetable"),
    ("veg_02", "carrot",    "vegetable"),
    ("veg_03", "broccoli",  "vegetable"),
    ("veg_04", "spinach",   "vegetable"),
    ("veg_05", "cucumber",  "vegetable"),
    ("veg_06", "zucchini",  "vegetable"),
    ("veg_07", "eggplant",  "vegetable"),
    ("veg_08", "bell pepper","vegetable"),
    ("veg_09", "celery",    "vegetable"),
    ("veg_10", "asparagus", "vegetable"),
    ("veg_11", "artichoke", "vegetable"),
    ("veg_12", "cauliflower","vegetable"),
    ("veg_13", "kale",      "vegetable"),
    ("veg_14", "radish",    "vegetable"),
    ("veg_15", "turnip",    "vegetable"),

    # ── Root vegetables / tubers (12) ────────────────────────────────────────
    ("root_01", "potato",       "root_veg"),
    ("root_02", "sweet potato", "root_veg"),
    ("root_03", "yam",          "root_veg"),
    ("root_04", "beet",         "root_veg"),
    ("root_05", "parsnip",      "root_veg"),
    ("root_06", "rutabaga",     "root_veg"),
    ("root_07", "ginger",       "root_veg"),
    ("root_08", "turmeric",     "root_veg"),
    ("root_09", "taro",         "root_veg"),
    ("root_10", "cassava",      "root_veg"),
    ("root_11", "jicama",       "root_veg"),
    ("root_12", "horseradish",  "root_veg"),

    # ── Leafy greens (12) ────────────────────────────────────────────────────
    ("leaf_01", "lettuce",      "leafy_green"),
    ("leaf_02", "arugula",      "leafy_green"),
    ("leaf_03", "watercress",   "leafy_green"),
    ("leaf_04", "chard",        "leafy_green"),
    ("leaf_05", "collard greens","leafy_green"),
    ("leaf_06", "endive",       "leafy_green"),
    ("leaf_07", "radicchio",    "leafy_green"),
    ("leaf_08", "romaine",      "leafy_green"),
    ("leaf_09", "bok choy",     "leafy_green"),
    ("leaf_10", "mustard greens","leafy_green"),
    ("leaf_11", "sorrel",       "leafy_green"),
    ("leaf_12", "mizuna",       "leafy_green"),

    # ── Herbs & spices (15) ──────────────────────────────────────────────────
    ("herb_01", "basil",      "herb"),
    ("herb_02", "oregano",    "herb"),
    ("herb_03", "thyme",      "herb"),
    ("herb_04", "rosemary",   "herb"),
    ("herb_05", "cilantro",   "herb"),
    ("herb_06", "parsley",    "herb"),
    ("herb_07", "dill",       "herb"),
    ("herb_08", "mint",       "herb"),
    ("herb_09", "sage",       "herb"),
    ("herb_10", "tarragon",   "herb"),
    ("herb_11", "chive",      "herb"),
    ("herb_12", "lavender",   "herb"),
    ("herb_13", "cinnamon",   "herb"),
    ("herb_14", "cumin",      "herb"),
    ("herb_15", "paprika",    "herb"),

    # ── Nuts & seeds (12) ────────────────────────────────────────────────────
    ("nut_01", "almond",      "nut"),
    ("nut_02", "walnut",      "nut"),
    ("nut_03", "pecan",       "nut"),
    ("nut_04", "cashew",      "nut"),
    ("nut_05", "pistachio",   "nut"),
    ("nut_06", "hazelnut",    "nut"),
    ("nut_07", "macadamia",   "nut"),
    ("nut_08", "peanut",      "nut"),
    ("nut_09", "chestnut",    "nut"),
    ("nut_10", "pine nut",    "nut"),
    ("nut_11", "sunflower seed","nut"),
    ("nut_12", "pumpkin seed", "nut"),

    # ── Grains & legumes (12) ────────────────────────────────────────────────
    ("grain_01", "wheat",     "grain"),
    ("grain_02", "rice",      "grain"),
    ("grain_03", "oat",       "grain"),
    ("grain_04", "barley",    "grain"),
    ("grain_05", "quinoa",    "grain"),
    ("grain_06", "corn",      "grain"),
    ("grain_07", "millet",    "grain"),
    ("grain_08", "buckwheat", "grain"),
    ("grain_09", "lentil",    "grain"),
    ("grain_10", "chickpea",  "grain"),
    ("grain_11", "black bean","grain"),
    ("grain_12", "soybean",   "grain"),

    # ── Seafood (12) ─────────────────────────────────────────────────────────
    ("sea_01", "salmon",   "seafood"),
    ("sea_02", "tuna",     "seafood"),
    ("sea_03", "shrimp",   "seafood"),
    ("sea_04", "lobster",  "seafood"),
    ("sea_05", "crab",     "seafood"),
    ("sea_06", "oyster",   "seafood"),
    ("sea_07", "mussel",   "seafood"),
    ("sea_08", "scallop",  "seafood"),
    ("sea_09", "sardine",  "seafood"),
    ("sea_10", "anchovy",  "seafood"),
    ("sea_11", "mackerel", "seafood"),
    ("sea_12", "squid",    "seafood"),

    # ── Dairy (10) ───────────────────────────────────────────────────────────
    ("dairy_01", "milk",        "dairy"),
    ("dairy_02", "butter",      "dairy"),
    ("dairy_03", "cheese",      "dairy"),
    ("dairy_04", "yogurt",      "dairy"),
    ("dairy_05", "cream",       "dairy"),
    ("dairy_06", "mozzarella",  "dairy"),
    ("dairy_07", "cheddar",     "dairy"),
    ("dairy_08", "parmesan",    "dairy"),
    ("dairy_09", "gouda",       "dairy"),
    ("dairy_10", "brie",        "dairy"),

    # ── Meat (10) ────────────────────────────────────────────────────────────
    ("meat_01", "chicken",  "meat"),
    ("meat_02", "beef",     "meat"),
    ("meat_03", "pork",     "meat"),
    ("meat_04", "lamb",     "meat"),
    ("meat_05", "turkey",   "meat"),
    ("meat_06", "duck",     "meat"),
    ("meat_07", "venison",  "meat"),
    ("meat_08", "bison",    "meat"),
    ("meat_09", "rabbit",   "meat"),
    ("meat_10", "bacon",    "meat"),

    # ── Phrases: cross-domain / ambiguous (20) ───────────────────────────────
    # Each phrase shares a word with a food item but has different semantics
    ("phrase_01", "freshly squeezed orange juice",          "phrase"),
    ("phrase_02", "tomato is technically a fruit",          "phrase"),
    ("phrase_03", "grandmother's apple pie recipe",         "phrase"),
    ("phrase_04", "tropical fruit salad with coconut",      "phrase"),
    ("phrase_05", "cherry blossom trees in spring",         "phrase"),
    ("phrase_06", "mandarin duck at the pond",              "phrase"),
    ("phrase_07", "carrot cake with cream cheese frosting", "phrase"),
    ("phrase_08", "lime green sports car",                  "phrase"),
    ("phrase_09", "banana republic clothing store",         "phrase"),
    ("phrase_10", "pineapple on pizza debate",              "phrase"),
    ("phrase_11", "sage advice from a wise mentor",         "phrase"),
    ("phrase_12", "mint condition vintage guitar",          "phrase"),
    ("phrase_13", "turkey the country in southeastern Europe","phrase"),
    ("phrase_14", "duck under the low doorway",             "phrase"),
    ("phrase_15", "crab mentality in workplace culture",    "phrase"),
    ("phrase_16", "peach colored sunset over the ocean",    "phrase"),
    ("phrase_17", "plum job at a prestigious company",      "phrase"),
    ("phrase_18", "rice paper lanterns at the festival",     "phrase"),
    ("phrase_19", "almond shaped eyes in portrait painting","phrase"),
    ("phrase_20", "chestnut horse galloping through meadow","phrase"),
]

# ── Resolve Ollama host ──────────────────────────────────────────────────────

ollama_host = get(ENV, "OLLAMA_HOST", nothing)
if ollama_host === nothing
    error("OLLAMA_HOST environment variable is required (e.g. OLLAMA_HOST=<host>:<port>)")
end

base_url = "http://$(ollama_host)"
embed_url = "$(base_url)/api/embed"

println("=== Embedding Generation ===")
println("Model:    $MODEL")
println("Endpoint: $embed_url")
println("Terms:    $(length(DATASET))")
println()

# ── Generate embeddings ──────────────────────────────────────────────────────

entries = []
dimension = nothing

for (i, (id, text, cluster)) in enumerate(DATASET)
    print("  [$i/$(length(DATASET))] $text ... ")

    body = Dict("model" => MODEL, "input" => text)
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
        @assert length(embedding) == dimension "Dimension mismatch: expected $dimension, got $(length(embedding))"
        println("ok")
    end

    push!(entries, Dict(
        "id"        => id,
        "text"      => text,
        "cluster"   => cluster,
        "embedding" => embedding,
    ))
end

# ── Write output ─────────────────────────────────────────────────────────────

output_path = joinpath(@__DIR__, "..", "test_data", "real_embeddings.json")
mkpath(dirname(output_path))

result = Dict(
    "model"     => MODEL,
    "dimension" => dimension,
    "entries"   => entries,
)

open(output_path, "w") do io
    JSON3.pretty(io, result)
end

println()
println("Written $(length(entries)) embeddings (dim=$dimension) to $output_path")
