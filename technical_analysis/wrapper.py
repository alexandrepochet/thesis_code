import technical_analysis.volume as v
import technical_analysis.volatility as volatility
import technical_analysis.trend as trend
import technical_analysis.momentum as momentum
import technical_analysis.others as others


def add_volume_ta(df, high, low, close, volume, fillna=False):
    """
    Add volume technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['volume_adi'] = v.acc_dist_index(df[high], df[low], df[close],
                                    df[volume], fillna=fillna)
    df['volume_obv'] = v.on_balance_volume(df[close], df[volume], fillna=fillna)
    df['volume_obvm'] = v.on_balance_volume_mean(df[close], df[volume], 10,
                                    fillna=fillna)
    df['volume_cmf'] = v.chaikin_money_flow(df[high], df[low], df[close],
                                        df[volume], fillna=fillna)
    df['volume_fi'] = v.force_index(df[close], df[volume], fillna=fillna)
    df['volume_em'] = v.ease_of_movement(df[high], df[low], df[close],
                                        df[volume], 14, fillna=fillna)
    df['volume_vpt'] = v.volume_price_trend(df[close], df[volume], fillna=fillna)
    df['volume_nvi'] = v.negative_volume_index(df[close], df[volume], fillna=fillna)
    
    return df

def add_volatility_ta(df, high, low, close, fillna=False):
    """
    Add volatility technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['volatility_atr'] = volatility.average_true_range(df[high], df[low], df[close],
                                                n=14, fillna=fillna)

    df['volatility_bbh'] = volatility.bollinger_hband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbl'] = volatility.bollinger_lband(df[close], n=20, ndev=2, fillna=fillna)
    df['volatility_bbm'] = volatility.bollinger_mavg(df[close], n=20, fillna=fillna)
    df['volatility_bbhi'] = volatility.bollinger_hband_indicator(df[close], n=20, ndev=2,
                                                    fillna=fillna)
    df['volatility_bbli'] = volatility.bollinger_lband_indicator(df[close], n=20, ndev=2,
                                                    fillna=fillna)

    df['volatility_kcc'] = volatility.keltner_channel_central(df[high], df[low], df[close],
                                                    n=10, fillna=fillna)
    df['volatility_kch'] = volatility.keltner_channel_hband(df[high], df[low], df[close],
                                                    n=10, fillna=fillna)
    df['volatility_kcl'] = volatility.keltner_channel_lband(df[high], df[low], df[close],
                                                    n=10, fillna=fillna)
    df['volatility_kchi'] = volatility.keltner_channel_hband_indicator(df[high], df[low],
                                                df[close], n=10, fillna=fillna)
    df['volatility_kcli'] = volatility.keltner_channel_lband_indicator(df[high], df[low],
                                                df[close], n=10, fillna=fillna)

    df['volatility_dch'] = volatility.donchian_channel_hband(df[close], n=20, fillna=fillna)
    df['volatility_dcl'] = volatility.donchian_channel_lband(df[close], n=20, fillna=fillna)
    df['volatility_dchi'] = volatility.donchian_channel_hband_indicator(df[close], n=20,
                                                            fillna=fillna)
    df['volatility_dcli'] = volatility.donchian_channel_lband_indicator(df[close], n=20,
                                                            fillna=fillna)
    
    return df

def add_trend_ta(df, high, low, close, fillna=False):
    """
    Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['trend_macd'] = trend.macd(df[close], n_fast=12, n_slow=26, fillna=fillna)
    df['trend_macd_signal'] = trend.macd_signal(df[close], n_fast=12, n_slow=26, n_sign=9,
                                    fillna=fillna)
    df['trend_macd_diff'] = trend.macd_diff(df[close], n_fast=12, n_slow=26, n_sign=9,
                                    fillna=fillna)
    df['trend_ema_fast'] = trend.ema_indicator(df[close], n=12, fillna=fillna)
    df['trend_ema_slow'] = trend.ema_indicator(df[close], n=26, fillna=fillna)
    df['trend_adx'] = trend.adx(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_adx_pos'] = trend.adx_pos(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_adx_neg'] = trend.adx_neg(df[high], df[low], df[close], n=14, fillna=fillna)
    df['trend_vortex_ind_pos'] = trend.vortex_indicator_pos(df[high], df[low], df[close], n=14,
                                    fillna=fillna)
    df['trend_vortex_ind_neg'] = trend.vortex_indicator_neg(df[high], df[low], df[close], n=14,
                                    fillna=fillna)
    df['trend_vortex_diff'] = abs(df['trend_vortex_ind_pos'] - df['trend_vortex_ind_neg'])
    df['trend_trix'] = trend.trix(df[close], n=15, fillna=fillna)
    df['trend_mass_index'] = trend.mass_index(df[high], df[low], n=9, n2=25, fillna=fillna)
    df['trend_cci'] = trend.cci(df[high], df[low], df[close], n=20, c=0.015,
                                    fillna=fillna)
    df['trend_dpo'] = trend.dpo(df[close], n=20, fillna=fillna)
    df['trend_kst'] = trend.kst(df[close], r1=10, r2=15, r3=20, r4=30, n1=10,
                            n2=10, n3=10, n4=15, fillna=fillna)
    df['trend_kst_sig'] = trend.kst_sig(df[close], r1=10, r2=15, r3=20, r4=30, n1=10,
                            n2=10, n3=10, n4=15, nsig=9, fillna=fillna)
    df['trend_kst_diff'] = df['trend_kst'] - df['trend_kst_sig']
    df['trend_ichimoku_a'] = trend.ichimoku_a(df[high], df[low], n1=9, n2=26, fillna=fillna)
    df['trend_ichimoku_b'] = trend.ichimoku_b(df[high], df[low], n2=26, n3=52, fillna=fillna)
    df['trend_visual_ichimoku_a'] = trend.ichimoku_a(df[high], df[low], n1=9, n2=26, visual=True, fillna=fillna)
    df['trend_visual_ichimoku_b'] = trend.ichimoku_b(df[high], df[low], n2=26, n3=52, visual=True, fillna=fillna)
    df['trend_aroon_up'] = trend.aroon_up(df[close], n=25, fillna=fillna)
    df['trend_aroon_down'] = trend.aroon_down(df[close], n=25, fillna=fillna)
    df['trend_aroon_ind'] = df['trend_aroon_up'] - df['trend_aroon_down']

    return df

def add_momentum_ta(df, high, low, close, volume, fillna=False):
    """Add trend technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['momentum_rsi'] = momentum.rsi(df[close], n=14, fillna=fillna)
    df['momentum_mfi'] = momentum.money_flow_index(df[high], df[low], df[close],
                                        df[volume], n=14, fillna=fillna)
    df['momentum_tsi'] = momentum.tsi(df[close], r=25, s=13, fillna=fillna)
    df['momentum_uo'] = momentum.uo(df[high], df[low], df[close], fillna=fillna)
    df['momentum_stoch'] = momentum.stoch(df[high], df[low], df[close], fillna=fillna)
    df['momentum_stoch_signal'] = momentum.stoch_signal(df[high], df[low], df[close], fillna=fillna)
    df['momentum_wr'] = momentum.wr(df[high], df[low], df[close], fillna=fillna)
    #df['momentum_ao'] = momentum.ao(df[high], df[low], fillna=fillna)
    return df

def add_others_ta(df, close, fillna=False):
    """Add others analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        close (str): Name of 'close' column.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df['others_dr'] = others.daily_return(df[close], fillna=fillna)
    df['others_dlr'] = others.daily_log_return(df[close], fillna=fillna)
    df['others_cr'] = others.cumulative_return(df[close], fillna=fillna)
    return df

def add_all_ta_features(df, open, high, low, close, volume, fillna=False):
    """Add all technical analysis features to dataframe.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe base.
        open (str): Name of 'open' column.
        high (str): Name of 'high' column.
        low (str): Name of 'low' column.
        close (str): Name of 'close' column.
        volume (str): Name of 'volume' column.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.core.frame.DataFrame: Dataframe with new features.
    """
    df = add_volume_ta(df, high, low, close, volume, fillna=fillna)
    df = add_volatility_ta(df, high, low, close, fillna=fillna)
    df = add_trend_ta(df, high, low, close, fillna=fillna)
    df = add_momentum_ta(df, high, low, close, volume, fillna=fillna)
    df = add_others_ta(df, close, fillna=fillna)
    
    return df