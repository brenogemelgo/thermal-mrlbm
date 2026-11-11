#pragma once

#include "output.cuh"
#include "reconstruction.cuh"
#include <array>
#include <limits>

#if defined(CASE_RAYLEIGH_BENARD)
namespace rbc
{
// Per-plane spatial moments; no persistent full-field statistics.
enum Stat
{
    T, U, V, W,
    T2, U2, V2, W2,
    WT, COND, KE, EU, ET,
    ETA, ETA_MIN, UMAX, RHO_SUM, BAD,
    NS
};
constexpr int THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS = THREADS / WARP_SIZE;
constexpr natural_t TILES = (STRIDE + THREADS - 1) / THREADS;
constexpr size_t PROFILE_SIZE = NS * NZ;
constexpr natural_t PLANE_BLOCKS = (PROFILE_SIZE + WARPS - 1) / WARPS;

template <int Field, typename Input>
__device__ __forceinline__ double primitive(
    Input moments,
    int x,
    int y,
    int z,
    natural_t step)
{
    x = (x + NX) % NX;
    y = (y + NY) % NY;
    const natural_t index = global3(x, y, z);
    double value = loadMoment(moments, index, Field);

    if constexpr (Field >= UX && Field <= UZ)
    {
        real_t acceleration[3];
        nodeAcceleration(loadMoment(moments, index, TEMP), x, y, z, step, acceleration);
        value -= double(acceleration[Field - UX]) / 2;
    }

    return value;
}

template <int Field, int Axis, typename Input>
__device__ __forceinline__ double derivative(
    Input moments,
    int x,
    int y,
    int z,
    natural_t step)
{
    if constexpr (Axis == 0)
    {
        return (primitive<Field>(moments, x + 1, y, z, step) - primitive<Field>(moments, x - 1, y, z, step)) / 2;
    }
    else if constexpr (Axis == 1)
    {
        return (primitive<Field>(moments, x, y + 1, z, step) - primitive<Field>(moments, x, y - 1, z, step)) / 2;
    }
    else
    {
        if (z == 0)
        {
            return (-3 * primitive<Field>(moments, x, y, z, step) + 4 * primitive<Field>(moments, x, y, 1, step) - primitive<Field>(moments, x, y, 2, step)) / 2;
        }
        if (z == NZ - 1)
        {
            return (3 * primitive<Field>(moments, x, y, z, step) - 4 * primitive<Field>(moments, x, y, z - 1, step) + primitive<Field>(moments, x, y, z - 2, step)) / 2;
        }
        return (primitive<Field>(moments, x, y, z + 1, step) - primitive<Field>(moments, x, y, z - 1, step)) / 2;
    }
}

template <int Field>
__device__ __forceinline__ double combine(double a, double b)
{
    if constexpr (Field == ETA_MIN)
    {
        return fmin(a, b);
    }
    else if constexpr (Field == UMAX)
    {
        return fmax(a, b);
    }
    else
    {
        return a + b;
    }
}

template <int Field>
__device__ __forceinline__ double reduceWarp(double value)
{
    constexpr_for<0, 5>(
        [&](const auto D) noexcept
        {
            value = combine<Field>(value, __shfl_down_sync(0xffffffff, value, WARP_SIZE >> (D + 1)));
        });
    return value;
}

// Warp-contiguous XY tiles. All fields share one block-wide barrier; constexpr
// indices keep the per-thread moments and velocity gradient in registers.
template <typename Input>
__device__ __forceinline__ void reduceTilesBody(Input moments, double *partial, natural_t step, natural_t zBegin)
{
    __shared__ double shared[WARPS][NS];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int xy = blockIdx.x * THREADS + threadIdx.x;
    const int z = blockIdx.y + zBegin;
    double values[NS] = {};
    values[ETA_MIN] = INFINITY;

    if (xy < STRIDE)
    {
        const int x = xy % NX;
        const int y = xy / NX;
        const double t = primitive<TEMP>(moments, x, y, z, step);
        const double u = primitive<UX>(moments, x, y, z, step);
        const double v = primitive<UY>(moments, x, y, z, step);
        const double w = primitive<UZ>(moments, x, y, z, step);
        values[T] = t;
        values[U] = u;
        values[V] = v;
        values[W] = w;
        values[T2] = t * t;
        values[U2] = u * u;
        values[V2] = v * v;
        values[W2] = w * w;
        values[WT] = w * (t - T_REF);
        values[KE] = (u * u + v * v + w * w) / 2;

        double gradient[3][3];
        constexpr_for<0, 3>(
            [&](const auto A) noexcept
            {
                constexpr_for<0, 3>(
                    [&](const auto B) noexcept
                    {
                        gradient[A][B] = derivative<UX + A, B>(moments, x, y, z, step);
                    });
            });
        constexpr_for<0, 3>(
            [&](const auto A) noexcept
            {
                const double g = derivative<TEMP, A>(moments, x, y, z, step);
                values[ET] += double(THERMAL_DIFFUSIVITY) * g * g;
                if constexpr (A == 2)
                {
                    values[COND] = -double(THERMAL_DIFFUSIVITY) * g;
                }
                constexpr_for<0, 3>(
                    [&](const auto B) noexcept
                    {
                        const double strain = gradient[A][B] + gradient[B][A];
                        values[EU] += double(VISCOSITY) * strain * strain / 2;
                    });
            });

        values[ETA] = values[EU] > 0 ? sqrt(sqrt(double(VISCOSITY) * VISCOSITY * VISCOSITY / values[EU])) : INFINITY;
        values[ETA_MIN] = values[ETA];
        values[UMAX] = sqrt(u * u + v * v + w * w);
        values[RHO_SUM] = loadMoment(moments, global3(x, y, z), RHO);
        values[BAD] = !isfinite(t + u + v + w + values[EU] + values[ET] + values[RHO_SUM]) || values[RHO_SUM] <= 0;
    }

    constexpr_for<0, NS>(
        [&](const auto Field) noexcept
        {
            const double value = reduceWarp<Field>(values[Field]);
            if (lane == 0)
            {
                shared[warp][Field] = value;
            }
        });
    __syncthreads();

    if (threadIdx.x < NS)
    {
        const int field = threadIdx.x;
        double value = shared[0][field];
        constexpr_for<1, WARPS>(
            [&](const auto K) noexcept
            {
                const double other = shared[K][field];
                value = field == ETA_MIN ? fmin(value, other) : field == UMAX ? fmax(value, other) : value + other;
            });
        partial[(size_t(blockIdx.y) * NS + field) * TILES + blockIdx.x] = value;
    }
}

__global__ void reduceTiles(const real_t *moments, double *partial, natural_t step = 0)
{
    reduceTilesBody(moments, partial, step, 0);
}

// One warp per (plane, observable), eight profiles per block. There is no
// inter-warp communication or block synchronization in this small second stage.
__device__ __forceinline__ void reducePlanesBody(const double *partial, double *profiles, natural_t profileSize)
{
    const int index = blockIdx.x * WARPS + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (index >= profileSize)
    {
        return;
    }

    const int field = index % NS;
    const int z = index / NS;
    double value = field == ETA_MIN ? INFINITY : 0;
    for (int tile = lane; tile < TILES; tile += WARP_SIZE)
    {
        const double other = partial[(size_t(z) * NS + field) * TILES + tile];
        value = field == ETA_MIN ? fmin(value, other) : field == UMAX ? fmax(value, other) : value + other;
    }
    constexpr_for<0, 5>(
        [&](const auto D) noexcept
        {
            const double other = __shfl_down_sync(0xffffffff, value, WARP_SIZE >> (D + 1));
            value = field == ETA_MIN ? fmin(value, other) : field == UMAX ? fmax(value, other) : value + other;
        });
    if (lane == 0)
    {
        profiles[z * NS + field] = field == ETA_MIN || field == UMAX ? value : value / STRIDE;
    }
}

__global__ void reducePlanes(const double *partial, double *profiles)
{
    reducePlanesBody(partial, profiles, PROFILE_SIZE);
}

struct Statistics
{
    double *partial = nullptr;
    double *deviceProfile = nullptr;
    std::vector<double> profile = std::vector<double>(PROFILE_SIZE);
    std::vector<double> sum = std::vector<double>(PROFILE_SIZE);
    std::vector<double> comp = std::vector<double>(PROFILE_SIZE);
    std::vector<double> row;
    std::array<double, 5> mean{}, m2{};
    uint64_t count = 0, lastSample = 0;
    double etaMin = INFINITY;

    explicit Statistics(bool allocateDevice = true)
    {
        row.reserve(21);
        if (!allocateDevice)
        {
            return;
        }
        outputCheckCuda(cudaMalloc(&partial, size_t(TILES) * PROFILE_SIZE * sizeof(double)), "allocate reduction tiles");
        outputCheckCuda(cudaMalloc(&deviceProfile, PROFILE_SIZE * sizeof(double)), "allocate profiles");
    }

    ~Statistics()
    {
        cudaFree(partial);
        cudaFree(deviceProfile);
    }

    static const char *seriesHeader()
    {
        return "step,t_ff,averaging,nu_bottom,nu_top,nu_volume,nu_kinetic,nu_thermal,temperature,urms,vrms,wrms,trms,kinetic_energy,epsilon_u,epsilon_t,nu_z_range,delta_t_cells,eta_min,max_velocity,mass_density";
    }

    void initializeSeries(bool resume, natural_t step)
    {
        if (resume && !std::filesystem::exists(statisticsPath()))
        {
            throw std::runtime_error("missing HDF5 statistics history; migrate legacy CSV before resuming");
        }

        h5out::Handle file(h5out::file(statisticsPath()), H5Fclose);
        h5out::Handle group(h5out::group(file, "statistics"), H5Gclose);
        if (resume)
        {
            if (!h5out::exists(group, "timeseries"))
            {
                throw std::runtime_error("missing HDF5 time-series history");
            }
            h5out::truncate(group, "timeseries", step);
            h5out::Handle data(H5Dopen2(group, "timeseries", H5P_DEFAULT), H5Dclose);
            const auto dims = h5out::dimensions(data);
            const uint64_t expected = 1 + std::min<uint64_t>(step, ActiveCase::timeAverageStart) / ActiveCase::timeAverageStride + count;
            if (dims[1] != 21 || dims[0] != expected)
            {
                throw std::runtime_error("HDF5 history does not match committed checkpoint samples");
            }
        }
        else if (h5out::exists(group, "timeseries"))
        {
            throw std::runtime_error("statistics already exist; use --continue or another run ID");
        }

        h5out::truncate(group, "runtime", step);
        if (!h5out::exists(group, "timeseries"))
        {
            h5out::table(group, "timeseries", seriesHeader(), {}, 0, 21);
        }
        h5out::attribute(file, "schema", "thermal-mrlbm-hdf5-v1");
        h5out::flush(file);
    }

    double volume(int f, const std::vector<double> &p) const
    {
        double v = 0;
        for (natural_t z = 0; z < NZ; ++z)
        {
            v += (z == 0 || z == NZ - 1 ? 0.5 : 1) * p[z * NS + f];
        }
        return v / ActiveCase::H;
    }

    std::array<double, 5> nusselt(const std::vector<double> &p) const
    {
        const double h = ActiveCase::H, dt = ActiveCase::DELTA_T;
        const double k = THERMAL_DIFFUSIVITY, nu = VISCOSITY;
        return {
            p[COND] * h / (k * dt),
            p[(NZ - 1) * NS + COND] * h / (k * dt),
            1 + h * volume(WT, p) / (k * dt),
            ActiveCase::RA > 0 ? 1 + volume(EU, p) * h * h * h * h * ActiveCase::PR * ActiveCase::PR / (nu * nu * nu * ActiveCase::RA) : 1,
            volume(ET, p) * h * h / (k * dt * dt)};
    }

    void sample(const real_t *m, natural_t step)
    {
        reduceTiles<<<dim3(TILES, NZ), THREADS>>>(m, partial, step);
        reducePlanes<<<PLANE_BLOCKS, THREADS>>>(partial, deviceProfile);
        outputCheckCuda(cudaGetLastError(), "RBC reductions");
        outputCheckCuda(cudaMemcpy(profile.data(), deviceProfile, PROFILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost), "copy profiles");
        sampleProfile(step);
    }

    // The two-GPU backend assembles only reduced per-plane values here.
    void sampleProfile(natural_t step)
    {
        if (volume(BAD, profile) > 0)
        {
            throw std::runtime_error("nonfinite RBC state or nonpositive density");
        }

        const auto nu = nusselt(profile);
        const bool avg = step > ActiveCase::timeAverageStart;
        if (avg)
        {
            ++count;
            lastSample = step;
            for (size_t i = 0; i < sum.size(); ++i)
            {
                // Kahan accumulation, except extrema and eta=inf.
                if (i % NS == ETA_MIN)
                {
                    sum[i] = count == 1 ? profile[i] : std::min(sum[i], profile[i]);
                    continue;
                }
                if (i % NS == UMAX)
                {
                    sum[i] = std::max(sum[i], profile[i]);
                    continue;
                }
                if (!std::isfinite(profile[i]))
                {
                    sum[i] = profile[i];
                    comp[i] = 0;
                    continue;
                }
                if (!std::isfinite(sum[i]))
                {
                    continue;
                }

                const double y = profile[i] - comp[i], tmp = sum[i] + y;
                comp[i] = (tmp - sum[i]) - y;
                sum[i] = tmp;
            }
            for (int i = 0; i < 5; ++i)
            {
                const double d = nu[i] - mean[i];
                mean[i] += d / count;
                m2[i] += d * (nu[i] - mean[i]);
            }
        }

        double nzMin = INFINITY, nzMax = -INFINITY;
        double em = INFINITY, um = 0;
        for (natural_t z = 0; z < NZ; ++z)
        {
            const double nz = (profile[z * NS + WT] + profile[z * NS + COND]) * ActiveCase::H / THERMAL_DIFFUSIVITY;
            nzMin = std::min(nzMin, nz);
            nzMax = std::max(nzMax, nz);
            em = std::min(em, profile[z * NS + ETA_MIN]);
            um = std::max(um, profile[z * NS + UMAX]);
        }
        if (avg)
        {
            etaMin = std::min(etaMin, em);
        }

        const double tm = volume(T, profile);
        const double u = volume(U, profile), v = volume(V, profile), w = volume(W, profile);
        auto rms = [](double a, double b) { return std::sqrt(std::max(0.0, a - b * b)); };
        row = {double(step), step / ActiveCase::T_FREEFALL, double(avg)};
        row.insert(row.end(), nu.begin(), nu.end());
        for (double value : {
                 tm, rms(volume(U2, profile), u), rms(volume(V2, profile), v), rms(volume(W2, profile), w),
                 rms(volume(T2, profile), tm), volume(KE, profile), volume(EU, profile), volume(ET, profile),
                 nzMax - nzMin, ActiveCase::H / (nu[0] + nu[1]), em, um, volume(RHO_SUM, profile)})
        {
            row.push_back(value);
        }

        h5out::OutputTimer timer;
        h5out::Handle file(h5out::file(statisticsPath()), H5Fclose);
        h5out::Handle group(h5out::group(file, "statistics"), H5Gclose);
        h5out::append(group, "timeseries", seriesHeader(), row);
        h5out::flush(file);
    }

    void exportAverages() const
    {
        std::vector<double> rows, scalars;
        if (count)
        {
            std::vector<double> p = sum;
            for (size_t i = 0; i < p.size(); ++i)
            {
                if (i % NS != ETA_MIN && i % NS != UMAX)
                {
                    p[i] /= count;
                }
            }
            for (natural_t z = 0; z < NZ; ++z)
            {
                const double *a = &p[z * NS];
                rows.push_back(z);
                rows.push_back(z / double(ActiveCase::H));
                for (int f : {T, U, V, W})
                {
                    rows.push_back(a[f]);
                }
                for (auto pair : {std::pair<int, int>{T2, T}, {U2, U}, {V2, V}, {W2, W}})
                {
                    rows.push_back(std::sqrt(std::max(0.0, a[pair.first] - a[pair.second] * a[pair.second])));
                }
                for (double value : {
                         a[WT], a[COND], (a[WT] + a[COND]) * ActiveCase::H / THERMAL_DIFFUSIVITY,
                         a[KE], a[EU], a[ET], a[ETA], a[ETA_MIN], 1 / a[ETA_MIN]})
                {
                    rows.push_back(value);
                }
            }
            for (int i = 0; i < 5; ++i)
            {
                const double sd = count > 1 ? std::sqrt(m2[i] / (count - 1)) : 0;
                for (double value : {double(i), double(count), mean[i], sd, sd / std::sqrt(double(count))})
                {
                    scalars.push_back(value);
                }
            }
        }

        h5out::Handle file(h5out::file(statisticsPath()), H5Fclose);
        h5out::Handle group(h5out::group(file, "statistics"), H5Gclose);
        h5out::table(group, "profiles", "z,z_over_h,temperature,ux,uy,uz,trms,urms,vrms,wrms,convective_flux,conductive_flux,nu_z,kinetic_energy,epsilon_u,epsilon_t,eta_mean,eta_min,delta_over_eta_min", rows, count ? NZ : 0, 19);
        h5out::table(group, "convergence", "estimator,samples,mean,sample_stddev,naive_stderr", scalars, count ? 5 : 0, 5);
        h5out::attribute(group, "estimator_order", "bottom,top,volume,kinetic,thermal");
        h5out::attribute(group, "averaged_samples", std::to_string(count));
        h5out::attribute(group, "last_averaged_step", std::to_string(lastSample));
        h5out::flush(file);
        // Correlation-aware block errors remain a post-processing operation.
    }
};
}
#endif
