def debug():
    df_Sc = hierarchy_cross_sectional(
        df,
        cross_sectional_aggregations,
        sparse=True,
        name_bottom_timeseries=name_bottom_timeseries,
    )
    df_St = hierarchy_temporal(df, time_index, temporal_aggregations, sparse=True)

    aggregation_cols = list(
        dict.fromkeys([col for cols in cross_sectional_aggregations for col in cols])
    )
    df = df.drop(columns=["week", "year", "month", "day"])
    X, Xind, targets = create_forecast_set(
        df, df_Sc, aggregation_cols, time_index, target, forecast_day=0
    )
    for seed in range(n_seeds):
        exp_name = experiment["exp_name"]
        params = default_params.copy()
        forecast_seed, t_train_seed, t_predict_seed = exp_m5_globalall(
            X,
            Xind,
            targets,
            target,
            time_index,
            end_train,
            start_test,
            df_Sc,
            df_St,
            exp_name=exp_name,
            params=params,
            exp_folder=exp_folder,
            seed=seed,
        )
        # Apply reconciliation methods
        forecasts_test = forecast_seed.loc[:, start_test:]
        forecasts_methods, t_reconciliation_seed = apply_reconciliation_methods(
            forecasts_test,
            df_Sc,
            targets.loc[:, :end_train],
            forecast_seed.loc[:, :end_train],
            methods=["ols", "wls_struct", "wls_var", "mint_shrink", "erm"],
            positive=True,
            return_timing=True,
        )
        # Add result to result df
        dfc = pd.concat({f"{seed}": forecasts_methods}, names=["Seed"])
        df_result = pd.concat((df_result, dfc))
        # Add timings to timings df
        df_seed = pd.DataFrame(
            {"t_train": t_train_seed, "t_predict": t_predict_seed}, index=[seed]
        )
        df_reconciliation = pd.DataFrame(t_reconciliation_seed, index=[seed])
        df_result_timings = pd.concat(
            (df_result_timings, pd.concat((df_seed, df_reconciliation), axis=1))
        )


hessian, denominator, Sc, Scd, St, Std = prepare_HierarchicalLoss(
    n_bottom_timeseries=n_bottom_timeseries,
    n_bottom_timesteps=n_bottom_timesteps,
    df_Sc=df_Sc,
    df_St=None,
)
fobj = partial(
    HierarchicalLossObjective,
    hessian=hessian,
    n_bottom_timeseries=n_bottom_timeseries,
    n_bottom_timesteps=n_bottom_timesteps,
    Sc=Sc,
    Scd=Scd,
)
