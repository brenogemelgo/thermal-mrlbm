#pragma once

#include "helpers.cuh"

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__host__ [[nodiscard]] static inline constexpr bool isValidBoundaryTypeConst() noexcept
{
    return nodeTypeValue == WALL || nodeTypeValue == INLET;
}

__host__ static inline void boundaryVelocityHost(
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    ubx = static_cast<real_t>(0);
    uby = static_cast<real_t>(0);
    ubz = nodeType == INLET ? U_CHAR : static_cast<real_t>(0);
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

__host__ static inline void invertIRBCOutflowMatrix(
    real_t (&a)[IRBC_OUTFLOW_UNKNOWNS][IRBC_OUTFLOW_UNKNOWNS],
    real_t (&inv)[IRBC_OUTFLOW_UNKNOWNS][IRBC_OUTFLOW_UNKNOWNS]) noexcept
{
    for (natural_t row = 0; row < IRBC_OUTFLOW_UNKNOWNS; ++row)
    {
        for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
        {
            inv[row][col] = row == col ? static_cast<real_t>(1) : static_cast<real_t>(0);
        }
    }

    for (natural_t pivot = 0; pivot < IRBC_OUTFLOW_UNKNOWNS; ++pivot)
    {
        natural_t pivotRow = pivot;
        real_t pivotAbs = a[pivot][pivot] < static_cast<real_t>(0)
                              ? -a[pivot][pivot]
                              : a[pivot][pivot];

        for (natural_t row = pivot + 1; row < IRBC_OUTFLOW_UNKNOWNS; ++row)
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
            for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
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

        for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
        {
            a[pivot][col] *= invPivot;
            inv[pivot][col] *= invPivot;
        }

        for (natural_t row = 0; row < IRBC_OUTFLOW_UNKNOWNS; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const real_t factor = a[row][pivot];

            for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
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
                     OMEGA_SHEAR * meqH),

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[0],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[1],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[2],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    VelocitySet::as4() *
                    h[3],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    VelocitySet::as4() *
                    h[4],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
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

__host__ [[nodiscard]] static inline bool isValidIRBC7NodeType(
    const unsigned int nodeType) noexcept
{
    if ((nodeType & OUTFLOW_FLAG) == OUTFLOW_FLAG)
    {
        return false;
    }

    if ((nodeType & INLET_FLAG) == INLET_FLAG)
    {
        return nodeType == INLET;
    }

    return (nodeType & FACE_MASK) != 0u;
}

__host__ static inline void assembleIRBCInverseRuntime(
    const unsigned int nodeType,
    real_t (&invOut)[IRBC_UNKNOWNS][IRBC_UNKNOWNS]) noexcept
{
    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityHost(nodeType, ubx, uby, ubz);

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
            if (!isMissingDirectionRuntime<Q>(nodeType))
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
                     OMEGA_SHEAR * meqH),

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[0],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[1],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[2],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    VelocitySet::as4() *
                    h[3],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    VelocitySet::as4() *
                    h[4],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
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

__host__ static inline void assembleOutflowIRBCInverse(
    real_t (&invOut)[IRBC_OUTFLOW_UNKNOWNS][IRBC_OUTFLOW_UNKNOWNS]) noexcept
{
    real_t momentRows[6][IRBC_OUTFLOW_UNKNOWNS] = {};

    momentRows[0][0] = static_cast<real_t>(1);
    momentRows[1][1] = static_cast<real_t>(1);
    momentRows[2][2] = static_cast<real_t>(1);
    momentRows[3][3] = static_cast<real_t>(1);
    momentRows[4][4] = static_cast<real_t>(1);
    momentRows[5][5] = static_cast<real_t>(1);

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            if constexpr (!isMissingDirectionConst<OUTFLOW, Q>())
            {
                return;
            }

            const real_t h[6] = {
                VelocitySet::hxx<Q>(),
                VelocitySet::hyy<Q>(),
                VelocitySet::hzz<Q>(),
                VelocitySet::hxy<Q>(),
                VelocitySet::hxz<Q>(),
                VelocitySet::hyz<Q>()};

            const real_t coeff[IRBC_OUTFLOW_UNKNOWNS] = {
                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[0],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[1],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    static_cast<real_t>(0.5) *
                    VelocitySet::as4() *
                    h[2],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    VelocitySet::as4() *
                    h[3],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    VelocitySet::as4() *
                    h[4],

                VelocitySet::w<Q>() *
                    T_OMEGA_SHEAR *
                    VelocitySet::as4() *
                    h[5]};

#pragma unroll
            for (natural_t row = 0; row < 6; ++row)
            {
#pragma unroll
                for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
                {
                    momentRows[row][col] -= h[row] * coeff[col];
                }
            }
        });

    real_t matrix[IRBC_OUTFLOW_UNKNOWNS][IRBC_OUTFLOW_UNKNOWNS] = {};

#pragma unroll
    for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
    {
        matrix[0][col] = momentRows[0][col] - momentRows[2][col];
        matrix[1][col] = momentRows[1][col] - momentRows[2][col];
        matrix[2][col] = momentRows[3][col];
        matrix[3][col] = momentRows[4][col];
        matrix[4][col] = momentRows[5][col];
        matrix[5][col] = static_cast<real_t>(0);
    }

    matrix[5][0] = static_cast<real_t>(1);
    matrix[5][1] = static_cast<real_t>(1);
    matrix[5][2] = static_cast<real_t>(1);

    invertIRBCOutflowMatrix(matrix, invOut);
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
    real_t hostOutflowTable[IRBC_OUTFLOW_TABLE_STRIDE] = {};

    for (natural_t nodeType = 1u; nodeType < IRBC_TABLE_NODE_TYPES; ++nodeType)
    {
        if (!isValidIRBC7NodeType(nodeType))
        {
            continue;
        }

        real_t inv[IRBC_UNKNOWNS][IRBC_UNKNOWNS] = {};
        assembleIRBCInverseRuntime(nodeType, inv);

        const natural_t tableOffset = nodeType * IRBC_TABLE_STRIDE;

        for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
        {
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                hostTable[tableOffset + row * IRBC_UNKNOWNS + col] = inv[row][col];
            }
        }
    }

    real_t outflowInv[IRBC_OUTFLOW_UNKNOWNS][IRBC_OUTFLOW_UNKNOWNS] = {};
    assembleOutflowIRBCInverse(outflowInv);

#pragma unroll
    for (natural_t row = 0; row < IRBC_OUTFLOW_UNKNOWNS; ++row)
    {
#pragma unroll
        for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
        {
            hostOutflowTable[row * IRBC_OUTFLOW_UNKNOWNS + col] = outflowInv[row][col];
        }
    }

    const cudaError_t boundaryError = cudaMemcpyToSymbol(IRBC_INVERSE, hostTable, sizeof(hostTable));
    if (boundaryError != cudaSuccess)
    {
        return boundaryError;
    }

    return cudaMemcpyToSymbol(IRBC_OUTFLOW_INVERSE, hostOutflowTable, sizeof(hostOutflowTable));
}
