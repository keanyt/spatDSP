res = spatDSP.run_dsp(
    adata,
    sample_key="Donor ID",          # donor/section/sample identifier
    group_key="ADChangeGroup",      # group label per cell
    groups=(1.0, 2.0),              # group0, group1 (e.g., Healthy vs Lesional)
    k=272,
    cheb_order=125,
    taus=list(np.geomspace(0.01, 50.0, 51)),
    exact=None,                     # None=auto, True=exact, False=Monte Carlo
    n_perms=5000,                   # used if Monte Carlo
    device="cuda",                  # or "cpu"
)

# columns are T_obs, P_Spatial, FDR_Spatial, N_Perm, Perm_Mode
res.table.head()

# dict: sample_id -> DataFrame(index=genes, columns=taus)
curves_for_sample = res.curves_by_sample["H20.33.001"]