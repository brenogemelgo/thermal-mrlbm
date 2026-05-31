#pragma once

#include "helpers.cuh"

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__host__ [[nodiscard]] static inline constexpr bool isValidBoundaryTypeConst() noexcept
{
    if constexpr (nodeTypeValue == BULK)
    {
        return false;
    }
    else
    {
        return !(((nodeTypeValue & WEST) != 0u && (nodeTypeValue & EAST) != 0u) ||
                 ((nodeTypeValue & SOUTH) != 0u && (nodeTypeValue & NORTH) != 0u) ||
                 ((nodeTypeValue & BACK) != 0u && (nodeTypeValue & FRONT) != 0u));
    }
}

__host__ static inline void boundaryVelocityHost(
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    if ((nodeType & FRONT) == FRONT)
    {
        ubx = U_CHAR;
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
    else
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
}

__host__ static inline void invertIRBCMatrix(
    real_t (&a)[IRBC_UNKNOWNS][IRBC_UNKNOWNS],
    real_t (&inv)[IRBC_UNKNOWNS][IRBC_UNKNOWNS]) noexcept
{
    for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
    {
        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            inv[row][col] = row == col ? static_cast<real_t>(1) : static_cast<real_t>(0);
        }
    }

    for (natural_t pivot = 0; pivot < IRBC_UNKNOWNS; ++pivot)
    {
        natural_t pivotRow = pivot;
        real_t pivotAbs = a[pivot][pivot] < static_cast<real_t>(0)
                              ? -a[pivot][pivot]
                              : a[pivot][pivot];

        for (natural_t row = pivot + 1; row < IRBC_UNKNOWNS; ++row)
        {
            const real_t valueAbs = a[row][pivot] < static_cast<real_t>(0)
                                        ? -a[row][pivot]
                                        : a[row][pivot];

            if (valueAbs > pivotAbs)
            {
                pivotAbs = valueAbs;
                pivotRow = row;
            }
        }

        if (pivotRow != pivot)
        {
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                const real_t tmpA = a[pivot][col];
                a[pivot][col] = a[pivotRow][col];
                a[pivotRow][col] = tmpA;

                const real_t tmpInv = inv[pivot][col];
                inv[pivot][col] = inv[pivotRow][col];
                inv[pivotRow][col] = tmpInv;
            }
        }

        const real_t pivotValue = a[pivot][pivot];
        const real_t invPivot = static_cast<real_t>(1) / pivotValue;

        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            a[pivot][col] *= invPivot;
            inv[pivot][col] *= invPivot;
        }

        for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const real_t factor = a[row][pivot];

            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                a[row][col] -= factor * a[pivot][col];
                inv[row][col] -= factor * inv[pivot][col];
            }
        }
    }
}

template <unsigned int nodeTypeValue>
__host__ static inline void assembleIRBCInverse(
    real_t (&invOut)[IRBC_UNKNOWNS][IRBC_UNKNOWNS]) noexcept
{
    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityHost(nodeTypeValue, ubx, uby, ubz);

    const real_t ub2 = ubx * ubx + uby * uby + ubz * ubz;

    real_t density[IRBC_UNKNOWNS] = {
        static_cast<real_t>(1),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    real_t momentRows[6][IRBC_UNKNOWNS] = {};

    momentRows[0][1] = static_cast<real_t>(1);
    momentRows[1][2] = static_cast<real_t>(1);
    momentRows[2][3] = static_cast<real_t>(1);
    momentRows[3][4] = static_cast<real_t>(1);
    momentRows[4][5] = static_cast<real_t>(1);
    momentRows[5][6] = static_cast<real_t>(1);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            if constexpr (!isMissingDirectionConst<nodeTypeValue, Q>())
            {
                return;
            }

            const real_t cx = static_cast<real_t>(VelocitySet::cx<Q>());
            const real_t cy = static_cast<real_t>(VelocitySet::cy<Q>());
            const real_t cz = static_cast<real_t>(VelocitySet::cz<Q>());

            const real_t h[6] = {
                VelocitySet::hxx<Q>(),
                VelocitySet::hyy<Q>(),
                VelocitySet::hzz<Q>(),
                VelocitySet::hxy<Q>(),
                VelocitySet::hxz<Q>(),
                VelocitySet::hyz<Q>()};

            const real_t cu = ubx * cx + uby * cy + ubz * cz;

            const real_t meqH =
                static_cast<real_t>(0.5) * VelocitySet::as4() *
                    (ubx * ubx * h[0] +
                     uby * uby * h[1] +
                     ubz * ubz * h[2]) +
                VelocitySet::as4() *
                    (ubx * uby * h[3] +
                     ubx * ubz * h[4] +
                     uby * ubz * h[5]);

            const real_t coeff[IRBC_UNKNOWNS] = {
                VelocitySet::w<Q>() *
                    (static_cast<real_t>(1) +
                     VelocitySet::as2() * cu +
                     OMEGA * meqH),

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[0],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[1],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[2],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    VelocitySet::as4() *
                    h[3],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    VelocitySet::as4() *
                    h[4],

                VelocitySet::w<Q>() *
                    T_OMEGA *
                    VelocitySet::as4() *
                    h[5]};

#pragma unroll
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                density[col] -= coeff[col];
            }

#pragma unroll
            for (natural_t row = 0; row < 6; ++row)
            {
#pragma unroll
                for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
                {
                    momentRows[row][col] -= h[row] * coeff[col];
                }
            }
        });

    real_t matrix[IRBC_UNKNOWNS][IRBC_UNKNOWNS] = {};

#pragma unroll
    for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
    {
        matrix[0][col] = density[col];
        matrix[1][col] = momentRows[0][col] - momentRows[2][col];
        matrix[2][col] = momentRows[1][col] - momentRows[2][col];
        matrix[3][col] = momentRows[3][col];
        matrix[4][col] = momentRows[4][col];
        matrix[5][col] = momentRows[5][col];
        matrix[6][col] = static_cast<real_t>(0);
    }

    matrix[6][0] = -ub2;
    matrix[6][1] = static_cast<real_t>(1);
    matrix[6][2] = static_cast<real_t>(1);
    matrix[6][3] = static_cast<real_t>(1);

    invertIRBCMatrix(matrix, invOut);
}

template <unsigned int nodeTypeValue>
__host__ static inline void fillIRBCBoundaryTableEntry(
    real_t (&hostTable)[IRBC_TABLE_SIZE]) noexcept
{
    if constexpr (!isValidBoundaryTypeConst<nodeTypeValue>())
    {
        return;
    }
    else
    {
        real_t inv[IRBC_UNKNOWNS][IRBC_UNKNOWNS] = {};
        assembleIRBCInverse<nodeTypeValue>(inv);

        constexpr natural_t tableOffset =
            static_cast<natural_t>(nodeTypeValue) * IRBC_TABLE_STRIDE;

#pragma unroll
        for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
        {
#pragma unroll
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                hostTable[tableOffset + row * IRBC_UNKNOWNS + col] = inv[row][col];
            }
        }
    }
}

__host__ [[nodiscard]] static inline cudaError_t initIRBCBoundaryTables() noexcept
{
    real_t hostTable[IRBC_TABLE_SIZE] = {};

    constexpr_for<0, 64>(
        [&](const auto nodeTypeConst) noexcept
        {
            fillIRBCBoundaryTableEntry<nodeTypeConst>(hostTable);
        });

    return cudaMemcpyToSymbol(IRBC_INVERSE, hostTable, sizeof(hostTable));
}
