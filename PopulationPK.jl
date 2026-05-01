using Pumas
using PumasUtilities
using SummaryTables
using PharmaDatasets
using Dates
using Random
using DataFramesMeta
using CSV
using ReadStatTables
using CategoricalArrays
using Random
using AlgebraOfGraphics
using CairoMakie
using PairPlots
using Statistics
using StatsBase
using GLM: lm, @formula
using LinearAlgebra
using StatsPlots
using Distributions
using PairPlots, ColorSchemes
using LaTeXStrings
using PumasUtilities: latexify
using Latexify
using PrettyTables

pwd()
cd("/Users/sandeepkg/Desktop/SOPHAS/PopPK Exercise/Assignments 2026/") 
df = CSV.read("pain_relief.csv", DataFrame)
df.logconc = ifelse.(ismissing.(df.Conc) .| (df.Conc .<= 0), missing, log.(df.Conc))
Cmax = combine(groupby(df, :Subject), :Conc => (x -> maximum(skipmissing(x))) => :Cmax)
df = outerjoin(Cmax, df, on = :Subject)

df_active = select(df, :Subject, :Age, :Weight, :Dose, :Time, :Conc, :route, :cmt, :amt, :evid, :logconc, :Cmax)

df_zero = @rsubset df :Time > 0

df_clean = filter(row -> !ismissing(row.Conc), df)
mean_df = @by df_clean [:Dose, :Time] begin
    :mean_conc = mean(:Conc)
    :std_conc = std(:Conc)
    :sem_conc = std(:Conc) / sqrt(length(:Conc))
    :n_subjects = length(unique(:Subject))
end

rng = Random.MersenneTwister(1234)

table_one(df, [:Age, :Weight]; groupby = :Dose)

conc_summary = summarytable(
    df_zero,
    :Conc => "Concentration (ng/mL)",
    summary = [mean => "Mean", std => "σ", length => "n"],
    rows = :Time => "Time (hours)",
    cols = :Dose => "Dose group",
)

df.DoseLabel = categorical(string.(df.Dose) .* " mg", levels = ["5 mg", "20 mg", "80 mg"], ordered=true)
spaghetti_by_dose = data(df) *
                    mapping(:Time, :Conc,
                           color = :Subject => nonnumeric,
                           layout = :DoseLabel => nonnumeric) *
                    visual(Lines, linewidth = 1.5)

fig_sphaghetti_by_dose = draw(spaghetti_by_dose, legend = (; show = false),
                              axis = (xlabel = "Time (hours)", ylabel = "Concentration (mg/L)",
                              ), figure = (title = "Spaghetti Plot ofConcentration vs Time", subtitle = "Faceted by Dose groups")
)
display(fig_sphaghetti_by_dose)

cmax_df = combine(groupby(mean_df, :Dose)) do sdf
    idx = argmax(sdf.mean_conc)
    (
        Cmax = sdf.mean_conc[idx],
        Tmax = sdf.Time[idx]
    )
end
set_theme!(Theme(
    fontsize = 14,
    Axis = (
        xlabelsize = 16,
        ylabelsize = 16,
        xticklabelsize = 12,
        yticklabelsize = 12,
    )
))
publication_fig = Figure()
ax = Axis(publication_fig[1, 1],
    xlabel = "Time (h)",
    ylabel = "Plasma Concentration (mg/L)",
    title = "Mean Concentration–Time Profile by Dose with Standard Deviation Error Bars",
)
colors = Makie.wong_colors()

df_summary = combine(groupby(df_clean, [:Dose, :Time])) do sdf
    (
        mean_conc = mean(sdf.Conc),
        sd_conc   = std(sdf.Conc),
        n         = length(sdf.Conc)
    )
end

dose_groups = sort(unique(df_summary.Dose))
for (i, d) in enumerate(dose_groups)
    sdf = df_summary[df_summary.Dose .== d, :]

    lines!(ax, sdf.Time, sdf.mean_conc,
        color = colors[i],
        linewidth = 3,
        label = "Dose = $(d)"
    )

    band!(ax,
    collect(skipmissing(sdf.Time)),
    collect(skipmissing(sdf.mean_conc .- sdf.sd_conc)),
    collect(skipmissing(sdf.mean_conc .+ sdf.sd_conc)),
    color = (colors[i], 0.2)
)
end
for (i, d) in enumerate(dose_groups)
    row = cmax_df[cmax_df.Dose .== d, :]

    CairoMakie.scatter!(ax, row.Tmax, row.Cmax, color = colors[i], markersize = 10)

    CairoMakie.text!(ax,
        "Cmax",
        position = (row.Tmax[1], row.Cmax[1]),
        align = (:left, :bottom),
        fontsize = 14,
        color = :black
    )
end
tmax_all = maximum(df_summary.Time)
terminal_start = 0.8 * tmax_all

text!(ax,
    "Apparent terminal\n phase",
    position = (terminal_start, minimum(df_summary.mean_conc)),
    align = (:left, :bottom),
    fontsize = 14,
    color = :black
)
axislegend(ax, framevisible = false)

publication_figure = publication_fig
display(publication_figure)

nca_pop = read_nca(df, 
                   id = :Subject,
                   time = :Time,
                   observations = :Conc,
                   amt = :amt,
                   route = :route,
                   group = [:Dose]
)
nca_results = run_nca(nca_pop, sigdigits = 4)
nca_df = deepcopy(nca_results.reportdf)
params = [:cmax => "Maximum Concentration (mg/L)", :tmax => "Time to Maximum Concentration (hours)", :auclast => "Area Under the Curve (last) (mg/L/hr)", :aucinf_obs => "Area Under the Curve (infinite) (mg/L/hr)", :half_life => "Half Life (hours)", :cl_f_obs => "Clearance (L/hr)", :vz_f_obs => "Volume of Distribution (L)"]
summary_stat_table = table_one(nca_df, params; groupby = :Dose)
display(summary_stat_table)

power_model_cmax = DoseLinearityPowerModel(nca_results, :cmax, level = 0.90)
power_plot_cmax = power_model(power_model_cmax)
dose_norm_cmax = dose_vs_dose_normalized(nca_results, :cmax)
display(dose_norm_cmax)

power_model_auc = DoseLinearityPowerModel(nca_results, :aucinf_obs, level = 0.90)
power_plot_auc = power_model(power_model_auc)
dose_norm_auc = dose_vs_dose_normalized(nca_results, :aucinf_obs)
display(dose_norm_auc)

beta_table = DataFrame(coeftable(power_model_cmax))
auc_table = DataFrame(coeftable(power_model_auc))
dose_prop_table = DataFrame(
    Parameter = ["Cmax", "AUC"],
    Beta = [beta_table.Estimate[1], auc_table.Estimate[1]],
    Lower90CI = [beta_table[1, 3], auc_table[1, 3]],
    Upper90CI = [beta_table[1, 4], auc_table[1, 4]]
)

pretty_table(dose_prop_table)

mean_cl = round(mean(skipmissing(nca_df.cl_f_obs)), digits = 3)
println("Initial Clearance = ", mean_cl, " ml/min")

mean_vz = round(mean(skipmissing(nca_df.vz_f_obs)), digits = 3)
println("Initial Volume of Distribution = ", mean_vz, " Litres")

Ka_individual = 1.0 ./ nca_df.tmax
ka = round(mean(skipmissing(Ka_individual)), digits = 3)
println("Initial Absorption Rate constant = ", ka, " hr^-1")

df_model = @select df_active begin
    :ID = :Subject
    :TIME = :Time
    :EVID = :evid
    :DOSE = :amt
    :Dose_mg = :Dose
    :CMT = :cmt
    :dv = :Conc
    :ROUTE = :route
    :AGE = :Age
    :WT = :Weight
end

pop = read_pumas(
    df_model;
    id = :ID,
    time = :TIME,
    evid = :EVID,
    amt = :DOSE,
    cmt = :CMT,
    route = :ROUTE,
    covariates = [:AGE, :WT, :Dose_mg],
    observations = [:dv]
)

om_cl, om_vl, om_ka = 0.09, 0.09, 0.09

one_comp_ke_model = @model begin
    @metadata begin
        desc = "One Compartment Model - First Order Elimination"
        timeu = u"minute"
    end
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)
        tvvc ∈ RealDomain(lower = 0.001)
        σ ∈ RealDomain(lower = 0.001)
    end

    @pre begin
        Cl = tvcl
        Vc = tvvc
    end

    @dynamics begin
        Central' = -(Cl / Vc) * Central
    end

    @derived begin
        cp := @. Central / Vc
        dv ~ @. Normal(cp, σ)
    end
end

init_params_ke = (tvcl = mean_cl, tvvc = mean_vz, σ = 0.2)

one_comp_ka_model = @model begin
    @metadata begin
        desc = "One-compartment model with first-order absorption"
    end

    @param begin
        tvcl ∈ RealDomain(lower = 0.001)     
        tvvc ∈ RealDomain(lower = 0.001)      
        tvka ∈ RealDomain(lower = 0.001)     
        σ ∈ RealDomain(lower = 0.001)        
    end

    @pre begin
        CL = tvcl 
        Vc = tvvc 
        Ka = tvka 
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/Vc) * Central
    end
    
    @derived begin
        cp := @. (Central / Vc)
        dv ~ @. Normal(cp, abs(cp) * σ)
    end
end

init_params_ka = (
    tvcl = mean_cl,
    tvvc = mean_vz,
    tvka = ka,
    σ = 0.2
)

one_comp_model_iiv = @model begin
    @metadata begin
        desc = "One-compartment model with first-order absorption"
    end

    @param begin
        tvcl ∈ RealDomain(lower = 0.001)      
        tvvc ∈ RealDomain(lower = 0.001)     
        tvka ∈ RealDomain(lower = 0.001)     
        Ω ∈ PDiagDomain(3)              
        σ ∈ RealDomain(lower = 0.001)         
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        V = tvvc * exp(η[2])
        Ka = tvka * exp(η[3])
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V) * Central
    end
    
    @derived begin
        cp := @. Central / V
        dv ~ @. Normal(cp, abs(cp) * σ) 
    end
end

init_params_iiv = (
    tvcl = mean_cl,
    tvvc = mean_vz,
    tvka = ka,
    Ω = Diagonal([om_cl, om_vl, om_ka]), 
    σ = 0.2
)

one_comp_model_add_error = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)
        tvv ∈ RealDomain(lower = 0.001)
        tvka ∈ RealDomain(lower = 0.001)
        Ω ∈ PDiagDomain(3)
        σ_add ∈ RealDomain(lower = 0.001)   
    end

    @random begin
        η ~ MvNormal(Ω)
    end
    
    @pre begin
        CL = tvcl * exp(η[1])
        V = tvv * exp(η[2])
        Ka = tvka * exp(η[3])
    end

    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V) * Central
    end

    @derived begin
        cp := @. Central / V
        dv ~ @. Normal(cp, σ_add)
    end
end

init_params_add_error = (
    tvcl = mean_cl,
    tvv = mean_vz,
    tvka = ka,
    Ω = Diagonal([om_cl, om_vl, om_ka]),
    σ_add = 0.15
)

one_comp_model_prop_error = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)
        tvv ∈ RealDomain(lower = 0.001)
        tvka ∈ RealDomain(lower = 0.001)
        Ω ∈ PDiagDomain(3)
        σ_prop ∈ RealDomain(lower = 0.001)  
    end

    @random begin
        η ~ MvNormal(Ω)
    end
    
    @pre begin
        CL = tvcl * exp(η[1])
        V = tvv * exp(η[2])
        Ka = tvka * exp(η[3])
    end

    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V) * Central
    end

    @derived begin
        cp := @. Central / V
        dv ~ @. Normal(cp, (cp * σ_prop))
    end
end

init_params_prop_error = (
    tvcl = mean_cl,
    tvv = mean_vz,
    tvka = ka,
    Ω = Diagonal([om_cl, om_vl, om_ka]),
    σ_prop = 0.20
)

one_comp_model_combined_error = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)
        tvv ∈ RealDomain(lower = 0.001)
        tvka ∈ RealDomain(lower = 0.001)
        Ω ∈ PDiagDomain(3)
        σ_add ∈ RealDomain(lower = 0.001)
        σ_prop ∈ RealDomain(lower = 0.001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        V = tvv * exp(η[2])
        Ka = tvka * exp(η[3])
    end

    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V) * Central
    end

    @derived begin
        cp := @. Central / V
        dv ~ @. Normal(cp, sqrt(σ_add^2 + (cp * σ_prop)^2))
    end
end

init_params_2_error = (
    tvcl = mean_cl,
    tvv = mean_vz,
    tvka = ka,
    Ω = Diagonal([om_cl, om_vl, om_ka]),
    σ_add = 0.05,
    σ_prop = 0.1
)

fit_one_comp_ke = fit(one_comp_ke_model, pop, init_params_ke, NaivePooled())
fit_one_comp_ka = fit(one_comp_ka_model, pop, init_params_ka, NaivePooled())
fit_one_comp_iiv = fit(one_comp_model_iiv, pop, init_params_iiv, FOCE())
fit_one_comp_add_error = fit(one_comp_model_add_error, pop, init_params_add_error, FOCE())
fit_one_comp_prop_error = fit(one_comp_model_prop_error, pop, init_params_prop_error, FOCE())
fit_one_comp_2_error = fit(one_comp_model_combined_error, pop, init_params_2_error, FOCE())

sim_one_comp_iiv = simobs(one_comp_model_iiv, pop, init_params_iiv, rng = rng)
sim_plot_one_cmp_iiv = sim_plot(one_comp_model_iiv, sim_one_comp_iiv; observations = [:dv])
sim_iiv_df = DataFrame(sim_one_comp_iiv)
sim_iiv = data(sim_iiv_df) * mapping(:time, :dv, color = :id => "", ) * visual(Lines, linewidth = 2, alpha = 0.7)
sim_iiv_plot = draw(sim_iiv; axis = (xlabel = "Time (hr)", ylabel = "Concentratiob (mg/L)", title = "Simulation of One Compartment Model", subtitle = "Interindividual variability on Cl, V, Ka",), figure = (; fontsize = 16), legend = (position = :bottom, show = false,))
display(sim_iiv_plot)

two_comp_model_ka = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)      
        tvv1 ∈ RealDomain(lower = 0.001)      
        tvq ∈ RealDomain(lower = 0.001)       
        tvv2 ∈ RealDomain(lower = 0.001)      
        tvka ∈ RealDomain(lower = 0.001)                     
        σ ∈ RealDomain(lower = 0.001)         
    end
    
    @pre begin
        CL = tvcl 
        V1 = tvv1 
        Q = tvq 
        V2 = tvv2 
        Ka = tvka 
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V1) * Central - (Q/V1) * Central + (Q/V2) * Peripheral
        Peripheral' = (Q/V1) * Central - (Q/V2) * Peripheral
    end
    
    @derived begin
        cp := @. Central / V1
        dv ~ @. Normal(cp, abs(cp) * σ)
    end
end

init_param_two_comp_ka = (
    tvcl = mean_cl,
    tvv1 = mean_vz * 0.7,
    tvq = mean_cl * 0.5,
    tvv2 = mean_vz * 0.3,
    tvka = ka,
    σ = 0.2
)

two_comp_iiv_model = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)
        tvv1 ∈ RealDomain(lower = 0.001)
        tvq ∈ RealDomain(lower = 0.001)
        tvv2 ∈ RealDomain(lower = 0.001)
        tvka ∈ RealDomain(lower = 0.001)
        Ω ∈ PDiagDomain(5)
        σ ∈ RealDomain(lower = 0.001)
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end
    
    @pre begin
        CL = tvcl * exp(η[1])
        V1 = tvv1 * exp(η[2])
        Q = tvq * exp(η[3])
        V2 = tvv2 * exp(η[4])
        Ka = tvka * exp(η[5])
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V1) * Central - (Q/V1) * Central + (Q/V2) * Peripheral
        Peripheral' = (Q/V1) * Central - (Q/V2) * Peripheral
    end
    
    @derived begin
        cp := @. Central / V1
        dv ~ @. Normal(cp, abs(cp) * σ)
    end
end

init_param_two_iiv = (
    tvcl = mean_cl,
    tvv1 = mean_vz * 0.7,
    tvq = mean_cl * 0.5,
    tvv2 = mean_vz * 0.3,
    tvka = ka,
    Ω = Diagonal([0.09, 0.09, 0.09, 0.09, 0.09]),
    σ = 0.20
)

two_comp_comb_error_model = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)     
        tvv1 ∈ RealDomain(lower = 0.001)     
        tvq ∈ RealDomain(lower = 0.001)      
        tvv2 ∈ RealDomain(lower = 0.001)      
        tvka ∈ RealDomain(lower = 0.001)      
        Ω ∈ PDiagDomain(5)
        σ_add ∈ RealDomain(lower = 0.0001)
        σ_prop ∈ RealDomain(lower = 0.0001)
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = tvcl * exp(η[1])
        V1 = tvv1 * exp(η[2])
        Q = tvq * exp(η[3])
        V2 = tvv2 * exp(η[4])
        Ka = tvka * exp(η[5])
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V1) * Central - (Q/V1) * Central + (Q/V2) * Peripheral
        Peripheral' = (Q/V1) * Central - (Q/V2) * Peripheral
    end
    
    @derived begin
        cp := @. Central / V1
        dv ~ @. Normal(cp, sqrt(σ_add^2 + (cp * σ_prop)^2))
    end
end

init_param_two_comb_error = (
    tvcl = mean_cl,
    tvv1 = mean_vz * 0.7,
    tvq = mean_cl * 0.5,
    tvv2 = mean_vz * 0.3,
    tvka = ka,
    Ω = Diagonal([0.09, 0.09, 0.09, 0.09, 0.09]),
    σ_add = 0.05,
    σ_prop = 0.1
)

fit_two_ka = fit(two_comp_model_ka, pop, init_param_two_comp_ka, NaivePooled())
fit_two_iiv = fit(two_comp_iiv_model, pop, init_param_two_iiv, FOCE())
fit_two_comb_error = fit(two_comp_comb_error_model, pop, init_param_two_comb_error, FOCE())

sim_two_comp_iiv = simobs(two_comp_iiv_model, pop, init_param_two_iiv, rng = rng)
sim_two_comp_iiv_df = DataFrame(sim_two_comp_iiv)
sim_two_comp_iiv = data(sim_two_comp_iiv_df) * mapping(:time, :dv, color = :id => "", ) * visual(Lines, linewidth = 2, alpha = 0.7)
sim_two_comp_iiv_plot = draw(sim_two_comp_iiv; axis = (xlabel = "Time (hr)", ylabel = "Concentratiob (mg/L)", title = "Simulation of Two Compartment Model", subtitle = "Interindividual variability on CL, Vc, Ka",), figure = (; fontsize = 16), legend = (position = :bottom, show = false,))
display(sim_two_comp_iiv_plot)

two_comp_iiv_diag_v2 = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)     
        tvv1 ∈ RealDomain(lower = 0.001)      
        tvq ∈ RealDomain(lower = 0.001)     
        tvv2 ∈ RealDomain(lower = 0.001)      
        tvka ∈ RealDomain(lower = 0.001)      
        Ω ∈ PDiagDomain(4)
        σ ∈ RealDomain(lower = 0.001)
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end
    
    @pre begin
        CL = tvcl * exp(η[1])
        V1 = tvv1 * exp(η[2])
        Q = tvq * exp(η[3])
        V2 = tvv2 
        Ka = tvka * exp(η[4])
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V1) * Central - (Q/V1) * Central + (Q/V2) * Peripheral
        Peripheral' = (Q/V1) * Central - (Q/V2) * Peripheral
    end
    
    @derived begin
        cp := @. Central / V1
        dv ~ @. Normal(cp, abs(cp) * σ)
    end
end

init_param_two_iiv_diag_v2 = (
    tvcl = mean_cl,
    tvv1 = mean_vz * 0.7,
    tvq = mean_cl * 0.5,
    tvv2 = mean_vz * 0.3,
    tvka = ka,
    Ω = Diagonal([0.09, 0.09, 0.09, 0.09]),
    σ = 0.2
)

two_comp_iiv_diag_ka = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)      
        tvv1 ∈ RealDomain(lower = 0.001)      
        tvq ∈ RealDomain(lower = 0.001)     
        tvv2 ∈ RealDomain(lower = 0.001)     
        tvka ∈ RealDomain(lower = 0.001)     
        Ω ∈ PDiagDomain(4)
        σ ∈ RealDomain(lower = 0.001)
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end
    
    @pre begin
        CL = tvcl * exp(η[1])
        V1 = tvv1 * exp(η[2])
        Q = tvq * exp(η[3])
        V2 = tvv2 * exp(η[4])
        Ka = tvka
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V1) * Central - (Q/V1) * Central + (Q/V2) * Peripheral
        Peripheral' = (Q/V1) * Central - (Q/V2) * Peripheral
    end
    
    @derived begin
        cp := @. Central / V1
        dv ~ @. Normal(cp, abs(cp) * σ)
    end
end

init_param_two_iiv_diag_ka = (
    tvcl = mean_cl,
    tvv1 = mean_vz * 0.7,
    tvq = mean_cl * 0.5,
    tvv2 = mean_vz * 0.3,
    tvka = ka,
    Ω = Diagonal([0.09, 0.09, 0.09, 0.09]),
    σ = 0.2
)

two_comp_iiv_diag_q = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001)      
        tvv1 ∈ RealDomain(lower = 0.001)      
        tvq ∈ RealDomain(lower = 0.001)     
        tvv2 ∈ RealDomain(lower = 0.001)      
        tvka ∈ RealDomain(lower = 0.001)     
        Ω ∈ PDiagDomain(4)
        σ ∈ RealDomain(lower = 0.001)
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end
    
    @pre begin
        CL = tvcl * exp(η[1])
        V1 = tvv1 * exp(η[2])
        Q = tvq
        V2 = tvv2 * exp(η[3])
        Ka = tvka * exp(η[4])
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/V1) * Central - (Q/V1) * Central + (Q/V2) * Peripheral
        Peripheral' = (Q/V1) * Central - (Q/V2) * Peripheral
    end
    
    @derived begin
        cp := @. Central / V1
        dv ~ @. Normal(cp, abs(cp) * σ)
    end
end

init_param_two_iiv_diag_q = (
    tvcl = mean_cl,
    tvv1 = mean_vz * 0.7,
    tvq = mean_cl * 0.5,
    tvv2 = mean_vz * 0.3,
    tvka = ka,
    Ω = Diagonal([0.09, 0.09, 0.09, 0.09]),
    σ = 0.2
)

fit_two_iiv_diag_v2 = fit(two_comp_iiv_diag_v2, pop, init_param_two_iiv_diag_v2, FOCE())
fit_two_iiv_diag_q = fit(two_comp_iiv_diag_q, pop, init_param_two_iiv_diag_q, FOCE())
fit_two_iiv_diag_ka = fit(two_comp_iiv_diag_ka, pop, init_param_two_iiv_diag_ka, FOCE())

model_comparison = DataFrame(
    Model = ["One Compartment Model - Ke", "One Compartment Model - Ka", "One Compartment Model - IIV", "One Compartment Model - Additive error", "One Compartment Model - Proportional error", "One Compartment Model - Combined error", "Two Compartment Model - Ka", "Two Compartment Model - IIV", "Two Compartment Model - combined error", "Two Compartment Model - No IIV on V2", "Two Compartment Model - No IIV on Q", "Two Compartment Model - No IIV on Ka"],
    LogLikelihood = [
        loglikelihood(fit_one_comp_ke),
        loglikelihood(fit_one_comp_ka),
        loglikelihood(fit_one_comp_iiv),
        loglikelihood(fit_one_comp_add_error),
        loglikelihood(fit_one_comp_prop_error),
        loglikelihood(fit_one_comp_2_error),
        loglikelihood(fit_two_ka),
        loglikelihood(fit_two_iiv),
        loglikelihood(fit_two_comb_error),
        loglikelihood(fit_two_iiv_diag_v2),
        loglikelihood(fit_two_iiv_diag_q),
        loglikelihood(fit_two_iiv_diag_ka)
    ],
    AIC = [
        aic(fit_one_comp_ke),
        aic(fit_one_comp_ka),
        aic(fit_one_comp_iiv),
        aic(fit_one_comp_add_error),
        aic(fit_one_comp_prop_error),
        aic(fit_one_comp_2_error),
        aic(fit_two_ka),
        aic(fit_two_iiv),
        aic(fit_two_comb_error),
        aic(fit_two_iiv_diag_v2),
        aic(fit_two_iiv_diag_q),
        aic(fit_two_iiv_diag_ka)
    ],
    BIC = [
        bic(fit_one_comp_ke),
        bic(fit_one_comp_ka),
        bic(fit_one_comp_iiv),
        bic(fit_one_comp_add_error),
        bic(fit_one_comp_prop_error),
        bic(fit_one_comp_2_error),
        bic(fit_two_ka),
        bic(fit_two_iiv),
        bic(fit_two_comb_error),
        bic(fit_two_iiv_diag_v2),
        bic(fit_two_iiv_diag_q),
        bic(fit_two_iiv_diag_ka)
    ]
)

model_comparison.OFV = -2 .* model_comparison.LogLikelihood

best_AIC = minimum(model_comparison.AIC)
best_BIC = minimum(model_comparison.BIC)
best_OFV = minimum(model_comparison.OFV)

model_comparison.ΔAIC = model_comparison.AIC .- best_AIC
model_comparison.ΔBIC = model_comparison.BIC .- best_BIC
model_comparison.ΔOFV = model_comparison.OFV .- best_OFV

pretty_table(model_comparison; crop = :none)

final_model = @model begin
    @param begin
        tvcl ∈ RealDomain(lower = 0.001,  init = mean_cl)      
        tvvc ∈ RealDomain(lower = 0.001,  init = mean_vz * 0.7)      
        tvq ∈ RealDomain(lower = 0.001,  init = mean_cl * 0.5)     
        tvvp ∈ RealDomain(lower = 0.001,  init = mean_vz * 0.3)      
        tvka ∈ RealDomain(lower = 0.001,  init = ka)     
        Ω ∈ PDiagDomain(4, init = Diagonal([0.09, 0.09, 0.09, 0.09]))
        σ ∈ RealDomain(lower = 0.001,  init = 0.2)
    end
    
    @random begin
        η ~ MvNormal(Ω)
    end
    
    @pre begin
        CL = tvcl * exp(η[1])
        VC = tvvc * exp(η[2])
        Q = tvq * exp(η[3])
        VP = tvvp 
        Ka = tvka * exp(η[4])
    end
    
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL/VC) * Central - (Q/VC) * Central + (Q/VP) * Peripheral
        Peripheral' = (Q/VC) * Central - (Q/VP) * Peripheral
    end
    
    @derived begin
        cp := @. Central / VC
        dv ~ @. Normal(cp, abs(cp) * σ)
    end
end

blocks = [:pre, :dynamics] 

model_representation = Latexify.LaTeXString(
    join([latexify(final_model, block, index = :bracket, snakecase = true) for block in blocks], "\n\n"),
);

model_representation

CL  = 2.543      
V1  = 11.443     
Q   = 1.337      
V2  = 9.605      
Ka  = 5.005      

k10 = CL / V1
k12 = Q / V1
k21 = Q / V2

A = k10 + k12 + k21
B = k10 * k21

alpha = (A + sqrt(A^2 - 4B)) / 2
beta  = (A - sqrt(A^2 - 4B)) / 2

t_half_alpha = log(2) / alpha
t_half_beta  = log(2) / beta

Vss = V1 + V2

MRT = Vss / CL

Dose = 1.0
AUC = Dose / CL

println("k10 = ", round(k10, digits = 2), " hr^-1")
println("k12 = ", round(k12, digits = 2), " hr^-1")
println("k21 = ", round(k21, digits = 2), " hr^-1")

println("alpha = ", round(alpha, digits = 2)," hr^-1")
println("beta = ", round(beta, digits = 2), " hr^-1")

println("t_half (Distribution) = ", round(t_half_alpha, digits = 2), " hrs")
println("t_half (Terminal) = ", round(t_half_beta, digits = 2), " hrs")

println("Vss = ", round(Vss, digits = 2), " L")
println("MRT = ", round(MRT, digits =  2), " hr")

println("AUC (Dose=1 mg) = ", round(AUC, digits = 2), " mg*hr/L")

infer_diag_v2 = infer(fit_two_iiv_diag_v2)
coef_table_diag_v2_df = coefficients_table(fit_two_iiv_diag_v2, infer_diag_v2)
pretty_table(coef_table_diag_v2_df; crop = :none)

eta_shrink_two = ηshrinkage(fit_two_iiv_diag_v2)

epsilon_shrink_two = ϵshrinkage(fit_two_iiv_diag_v2)

final_fit = fit_two_iiv_diag_v2
final_inspect = inspect(final_fit; nsim = 200)
inspect_df = DataFrame(final_inspect)
df_insp = DataFrame(final_inspect)
gof_plots = goodness_of_fit(final_inspect; observations = [:dv])
gof = gof_plots.figure
Label(gof[0, :], "Two Compartment Model with IIV - Goodness of Fit \n No IIV on Vp ", font = :bold, fontsize = 18)
display(gof)

npde_plot = npde_dist(final_inspect; color = :brown, zeroline_color = :red, figure = (; fontsize = 18))
display(npde_plot)

fig_subject_fits2 = subject_fits(
    final_inspect;
    separate = true,
    paginate = true,
    facet = (; linkyaxes = false),
    figure = (; fontsize = 18),
    axis = (; xlabel = "Time (hr)", ylabel = "Concentration (mg/L)"),
)
fig_subject_fits2[12]

df_iwres = dropmissing(inspect_df, :dv_iwres)
outliers_iwres = filter(row -> abs(row.dv_iwres) > 3, df_iwres)
print(outliers_iwres)
df_npde = dropmissing(inspect_df, :dv_npde)
outliers_npde = filter(row -> abs(row.dv_npde) > 3, df_npde)
print(outliers_npde)

ebes = @chain df_insp begin
    select(r"^η")
    dropmissing
end

ebe_pairplot = pairplot(ebes => (PairPlots.Scatter(marker = '∘', markersize = 24, alpha = 0.3, color = ColorSchemes.tab10.colors[1]), 
                                 PairPlots.TrendLine(color = :red),         
                                 PairPlots.PearsonCorrelation(fontsize = 15, color = :black),
                                 PairPlots.MarginHist(color = ColorSchemes.tab10.colors[1])), 
                            fullgrid = false
)
display(ebe_pairplot)

eta_dist = empirical_bayes_dist(inspect(fit_two_iiv_diag_v2); color = :lightblue, zeroline_color = :red, figure = (; fontsize = 18))
display(eta_dist)

empirical_bayes_vs_covariates(inspect(fit_two_iiv_diag_v2), zeroline_color = :red)

full_vpc = vpc(fit_two_iiv_diag_v2, 1000; observations = [:dv])
full_vpc_plot = vpc_plot(two_comp_iiv_diag_v2, full_vpc;
    axis = (;
        xlabel = "Time (hr)",
        ylabel = "Observed/Predicted\n Drug Concentration (ng/mL)",
    ),
    facet = (; combinelabels = true),
)
display(full_vpc_plot)

vpc_stratify_dose = vpc(fit_two_iiv_diag_v2, 1000; observations = [:dv], stratify_by = [:Dose_mg])
vpc_stratify_dose_plot = vpc_plot(two_comp_iiv_diag_v2, vpc_stratify_dose;
    rows = 1,
    columns = 3,
    figure = (; size = (1400, 1000), fontsize = 22),
    axis = (;
        xlabel = "Time (hr)",
        ylabel = "Observed/Predicted\n Drug Concentration (ng/mL)",
    ),
    facet = (; combinelabels = true),
)
display(vpc_stratify_dose_plot)

blocks = [:param, :random, :pre, :dynamics, :derived] 

model_representation_full = Latexify.LaTeXString(
    join([latexify(two_comp_iiv_diag_v2, block, index = :bracket, snakecase = true) for block in blocks], "\n\n"),
);
model_representation_full