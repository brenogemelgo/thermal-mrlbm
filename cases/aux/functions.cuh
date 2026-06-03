#pragma once

#include <limits>

template <typename T>
[[nodiscard]] inline consteval T csqrt(T x)
{
    if (x < 0)
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == 0 || x == std::numeric_limits<T>::infinity())
    {
        return x;
    }

    T cur = x / 2.0;
    T prev = 0.0;

    while (cur != prev)
    {
        prev = cur;
        cur = 0.5 * (cur + x / cur);
    }

    return cur;
}
