using Printf
using NPZ
include("util.jl")
include("kmeans_filters.jl")
include("quantum_filters.jl")

log_file = open("run_filters.log", "a")

for name in ARGS
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
