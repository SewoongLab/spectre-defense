using Printf
using NPZ
include("util.jl")
include("kmeans_filters.jl")
include("quantum_filters.jl")

names = [
    "r32p-sgd-94-1xp1000",
    "r32p-sgd-94-1xp500",
    "r32p-sgd-94-1xp250",
    "r32p-sgd-94-1xp125",
    "r32p-sgd-94-1xp62",
    "r32p-sgd-94-1xp31",
    "r32p-sgd-94-1xp15",
    "r32p-sgd-94-2xp1000",
    "r32p-sgd-94-2xp500",
    "r32p-sgd-94-2xp250",
    "r32p-sgd-94-2xp125",
    "r32p-sgd-94-2xp62",
    "r32p-sgd-94-2xp31",
    "r32p-sgd-94-2xp15",
    "r32p-sgd-94-3xp1000",
    "r32p-sgd-94-3xp500",
    "r32p-sgd-94-3xp250",
    "r32p-sgd-94-3xp125",
    "r32p-sgd-94-3xp62",
    "r32p-sgd-94-3xp31",
    "r32p-sgd-94-3xp15",
    "r32p-sgd-94-1xs500",
    "r32p-sgd-94-1xs250",
    "r32p-sgd-94-1xs125",
    "r32p-sgd-94-1xs62",
    "r32p-sgd-94-1xs31",
    "r32p-sgd-94-1xs15",
    "r32p-sgd-94-2xs500",
    "r32p-sgd-94-2xs250",
    "r32p-sgd-94-2xs125",
    "r32p-sgd-94-2xs62",
    "r32p-sgd-94-2xs31",
    "r32p-sgd-94-2xs15",
    # "r32p-sgd-4-300l2AAcl500",
    # "r32p-sgd-4-300l2AAcl250",
    # "r32p-sgd-4-300l2AAcl125",
    # "r32p-sgd-4-300l2AAcl62",
    # "r32p-sgd-4-300l2AAcl31",
    # "r32p-sgd-4-300l2AAcl15",
    # "r32p-sgd-4-8linfAAcl500",
    # "r32p-sgd-4-8linfAAcl250",
    # "r32p-sgd-4-8linfAAcl125",
    # "r32p-sgd-4-8linfAAcl62",
    # "r32p-sgd-4-8linfAAcl31",
    # "r32p-sgd-4-8linfAAcl15",
    # "r32p-sgd-4-02ganAAcl500",
    # "r32p-sgd-4-02ganAAcl250",
    # "r32p-sgd-4-02ganAAcl125",
    # "r32p-sgd-4-02ganAAcl62",
    # "r32p-sgd-4-02ganAAcl31",
    # "r32p-sgd-4-02ganAAcl15",
    # "r18-ranger-53-1xp500",
    # "r18-ranger-53-1xp250",
    # "r18-ranger-53-1xp125",
    # "r18-ranger-53-1xp62",
    # "r18-ranger-53-1xp31",
    # "r18-ranger-53-1xp15",
    # "r18-ranger-53-2xp500",
    # "r18-ranger-53-2xp250",
    # "r18-ranger-53-2xp125",
    # "r18-ranger-53-2xp62",
    # "r18-ranger-53-2xp31",
    # "r18-ranger-53-2xp15",
    # "r18-ranger-53-3xp500",
    # "r18-ranger-53-3xp250",
    # "r18-ranger-53-3xp125",
    # "r18-ranger-53-3xp62",
    # "r18-ranger-53-3xp31",
    # "r18-ranger-53-3xp15",
    # "r18-ranger-53-1xs500",
    # "r18-ranger-53-1xs250",
    # "r18-ranger-53-1xs125",
    # "r18-ranger-53-1xs62",
    # "r18-ranger-53-1xs31",
    # "r18-ranger-53-1xs15",
    # "r18-ranger-53-2xs500",
    # "r18-ranger-53-2xs250",
    # "r18-ranger-53-2xs125",
    # "r18-ranger-53-2xs62",
    # "r18-ranger-53-2xs31",
    # "r18-ranger-53-2xs15",
    # "r32p-sgd-09-1xp500",
    # "r32p-sgd-09-1xp125",
    # "r32p-sgd-09-3xp500",
    # "r32p-sgd-09-3xp125",
    # "r32p-sgd-17-1xp500",
    # "r32p-sgd-17-1xp125",
    # "r32p-sgd-17-3xp500",
    # "r32p-sgd-17-3xp125",
    # "r32p-sgd-25-1xp500",
    # "r32p-sgd-25-1xp125",
    # "r32p-sgd-25-3xp500",
    # "r32p-sgd-25-3xp125",
    "r32p-sgd-38-1xp500",
    "r32p-sgd-38-1xp125",
    "r32p-sgd-38-3xp500",
    "r32p-sgd-38-3xp125",
    "r32p-sgd-41-1xp500",
    "r32p-sgd-41-1xp125",
    "r32p-sgd-41-3xp500",
    "r32p-sgd-41-3xp125",
    "r32p-sgd-53-1xp500",
    "r32p-sgd-53-1xp125",
    "r32p-sgd-53-3xp500",
    "r32p-sgd-53-3xp125",
    "r32p-sgd-62-1xp500",
    "r32p-sgd-62-1xp125",
    "r32p-sgd-62-3xp500",
    "r32p-sgd-62-3xp125",
    "r32p-sgd-70-1xp500",
    "r32p-sgd-70-1xp125",
    "r32p-sgd-70-3xp500",
    "r32p-sgd-70-3xp125",
    "r32p-sgd-86-1xp500",
    "r32p-sgd-86-1xp125",
    "r32p-sgd-86-3xp500",
    "r32p-sgd-86-3xp125",
]

log_file = open("run_filters.log", "a")

for name in names
    target_label = parse(Int, split(name, "-")[3][end:end])
    reps = npzread("output/$(name)/label_$(target_label)_reps.npy")'
    n = size(reps)[2]
    eps = parse(Int, match(r"[0-9]+$", name).match)
    removed = round(Int, 1.5*eps)

    @printf("%s: Running PCA filter\n", name)
    reps_pca, U = pca(reps, 1)
    pca_poison_ind = k_lowest_ind(-abs.(mean(reps_pca[1, :]) .- reps_pca[1, :]), round(Int, 1.5*eps))
    poison_removed = sum(pca_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @show poison_removed, clean_removed
    @printf(log_file, "%s-pca: %d, %d\n", name, poison_removed, clean_removed)
    npzwrite("output/$(name)/mask-pca-target.npy", pca_poison_ind)


    @printf("%s: Running kmeans filter\n", name)
    kmeans_poison_ind = .! kmeans_filter2(reps, eps)
    poison_removed = sum(kmeans_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @show poison_removed, clean_removed
    @printf(log_file, "%s-kmeans: %d, %d\n", name, poison_removed, clean_removed)
    npzwrite("output/$(name)/mask-kmeans-target.npy", kmeans_poison_ind)

    @printf("%s: Running quantum filter\n", name)
    quantum_poison_ind = .! rcov_auto_quantum_filter(reps, eps)
    poison_removed = sum(quantum_poison_ind[end-eps+1:end])
    clean_removed = removed - poison_removed
    @show poison_removed, clean_removed
    @printf(log_file, "%s-quantum: %d, %d\n", name, poison_removed, clean_removed)
    npzwrite("output/$(name)/mask-rcov-target.npy", quantum_poison_ind)
end
