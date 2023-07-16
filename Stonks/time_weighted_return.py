def calculate_time_weighted_return(*periods, percents: bool=False):
    twr = 1.
    if percents:
        for percent in periods:
            twr *= (1 + (percent / 100))
    else:
        for (initial_value, end_value) in periods:
            period_return = ((end_value - initial_value) / initial_value) + 1
            twr *= period_return
    twr -= 1

    return twr
