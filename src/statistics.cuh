#pragma once

#include "output.cuh"

#include <algorithm>
#include <array>
#include <limits>

struct JetStatisticsAccumulator
{
    std::vector<double> sumUx;
    std::vector<double> sumUy;
    std::vector<double> sumUz;
    std::vector<double> sumT;
    std::vector<double> sumUx2;
    std::vector<double> sumUy2;
    std::vector<double> sumUz2;
    std::vector<double> sumT2;
    std::vector<double> sumUzT;
    std::vector<real_t> fieldData;
    std::vector<real_t> uzPhysical;

    JetStatisticsAccumulator()
        : sumUx(CELLS, 0.0),
          sumUy(CELLS, 0.0),
          sumUz(CELLS, 0.0),
          sumT(CELLS, 0.0),
          sumUx2(CELLS, 0.0),
          sumUy2(CELLS, 0.0),
          sumUz2(CELLS, 0.0),
          sumT2(CELLS, 0.0),
          sumUzT(CELLS, 0.0),
          fieldData(CELLS),
          uzPhysical(CELLS)
    {
    }
};

static inline void statisticsCheckCuda(
    const cudaError_t err,
    const char *call)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA statistics error: " << call << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void copyStatisticsMoment(
    const real_t *deviceMoments,
    const natural_t moment,
    std::vector<real_t> &fieldData)
{
    statisticsCheckCuda(
        cudaMemcpy(fieldData.data(), deviceMoments + CELLS * moment, CELLS * sizeof(real_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy statistics field");
}

static inline void accumulateStatisticsVelocity(
    JetStatisticsAccumulator &statistics,
    const real_t *deviceMoments,
    const natural_t moment,
    std::vector<double> &sum,
    std::vector<double> &sum2,
    std::vector<real_t> *physicalScratch)
{
    copyStatisticsMoment(deviceMoments, moment, statistics.fieldData);

    constexpr double invScaleI = static_cast<double>(1) / static_cast<double>(VelocitySet::scaleI());
    for (natural_t idx = 0; idx < CELLS; ++idx)
    {
        const double value = static_cast<double>(statistics.fieldData[idx]) * invScaleI;
        sum[idx] += value;
        sum2[idx] += value * value;
        if (physicalScratch != nullptr)
        {
            (*physicalScratch)[idx] = static_cast<real_t>(value);
        }
    }
}

static inline void accumulateStatisticsSample(
    JetStatisticsAccumulator &statistics,
    const real_t *deviceMoments)
{
    accumulateStatisticsVelocity(statistics, deviceMoments, UX, statistics.sumUx, statistics.sumUx2, nullptr);
    accumulateStatisticsVelocity(statistics, deviceMoments, UY, statistics.sumUy, statistics.sumUy2, nullptr);
    accumulateStatisticsVelocity(statistics, deviceMoments, UZ, statistics.sumUz, statistics.sumUz2, &statistics.uzPhysical);

    copyStatisticsMoment(deviceMoments, TEMP, statistics.fieldData);
    for (natural_t idx = 0; idx < CELLS; ++idx)
    {
        const double temperature = static_cast<double>(statistics.fieldData[idx]);
        const double uz = static_cast<double>(statistics.uzPhysical[idx]);
        statistics.sumT[idx] += temperature;
        statistics.sumT2[idx] += temperature * temperature;
        statistics.sumUzT[idx] += uz * temperature;
    }
}

static inline void writeStatisticsMeanBinary(
    const std::vector<double> &sum,
    const natural_t statisticsSampleCount,
    const std::filesystem::path &path)
{
    if (statisticsSampleCount == 0)
    {
        std::cerr << "Cannot write statistics with zero samples" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        std::cerr << "Could not open statistics output: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    constexpr std::size_t chunkCells = 1u << 20;
    std::vector<double> chunk(std::min<std::size_t>(chunkCells, static_cast<std::size_t>(CELLS)));
    const double invSamples = static_cast<double>(1) / static_cast<double>(statisticsSampleCount);

    for (std::size_t offset = 0; offset < static_cast<std::size_t>(CELLS); offset += chunk.size())
    {
        const std::size_t count = std::min<std::size_t>(chunk.size(), static_cast<std::size_t>(CELLS) - offset);
        for (std::size_t i = 0; i < count; ++i)
        {
            chunk[i] = sum[offset + i] * invSamples;
        }

        out.write(reinterpret_cast<const char *>(chunk.data()), static_cast<std::streamsize>(count * sizeof(double)));
        if (!out)
        {
            std::cerr << "Could not write statistics output: " << path << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

static inline void writeStatisticsMetadata(
    const natural_t statisticsSampleCount,
    const natural_t sampleStartStep,
    const natural_t sampleEndStep,
    const std::filesystem::path &path)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Could not open statistics metadata output: " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    out << "{\n";
    out << "  \"NX\": " << NX << ",\n";
    out << "  \"NY\": " << NY << ",\n";
    out << "  \"NZ\": " << NZ << ",\n";
    out << "  \"DIAM\": " << static_cast<double>(static_cast<real_t>(2) * RADIUS) << ",\n";
    out << "  \"RADIUS\": " << static_cast<double>(RADIUS) << ",\n";
    out << "  \"L_CHAR\": " << static_cast<double>(L_CHAR) << ",\n";
    out << "  \"U_CHAR\": " << static_cast<double>(U_CHAR) << ",\n";
    out << "  \"T_REF\": " << static_cast<double>(T_REF) << ",\n";
    out << "  \"T_COLD\": " << static_cast<double>(T_COLD) << ",\n";
    out << "  \"T_HOT\": " << static_cast<double>(T_HOT) << ",\n";
    out << "  \"statisticsSampleCount\": " << statisticsSampleCount << ",\n";
    out << "  \"sampleStartStep\": " << sampleStartStep << ",\n";
    out << "  \"sampleEndStep\": " << sampleEndStep << ",\n";
    out << "  \"dtype\": \"float64\",\n";
    out << "  \"endianness\": \"little\",\n";
    out << "  \"layout\": \"idx = x + NX*(y + NY*z)\",\n";
    out << "  \"shape\": [\"NZ\", \"NY\", \"NX\"],\n";
    out << "  \"sampleTiming\": \"initial state at sampleStartStep, then every state after streamCollide/swap through sampleEndStep\",\n";
    out << "  \"velocityBasis\": \"physical/raw velocities: ux=UX/scaleI, uy=UY/scaleI, uz=UZ/scaleI; stored-basis UX/UY/UZ are not exported as validation velocities\",\n";
    out << "  \"temperatureBasis\": \"T=TEMP\",\n";
    out << "  \"fields\": [\"mean_ux\", \"mean_uy\", \"mean_uz\", \"mean_T\", \"mean_ux2\", \"mean_uy2\", \"mean_uz2\", \"mean_T2\", \"mean_uzT\"]\n";
    out << "}\n";
}

static inline void writeStatisticsOutput(
    const JetStatisticsAccumulator &statistics,
    const natural_t statisticsSampleCount,
    const natural_t sampleStartStep,
    const natural_t sampleEndStep)
{
    const std::filesystem::path dir = getSimulationOutputDirectory() / "statistics";
    std::filesystem::create_directories(dir);

    writeStatisticsMeanBinary(statistics.sumUx, statisticsSampleCount, dir / "mean_ux.bin");
    writeStatisticsMeanBinary(statistics.sumUy, statisticsSampleCount, dir / "mean_uy.bin");
    writeStatisticsMeanBinary(statistics.sumUz, statisticsSampleCount, dir / "mean_uz.bin");
    writeStatisticsMeanBinary(statistics.sumT, statisticsSampleCount, dir / "mean_T.bin");
    writeStatisticsMeanBinary(statistics.sumUx2, statisticsSampleCount, dir / "mean_ux2.bin");
    writeStatisticsMeanBinary(statistics.sumUy2, statisticsSampleCount, dir / "mean_uy2.bin");
    writeStatisticsMeanBinary(statistics.sumUz2, statisticsSampleCount, dir / "mean_uz2.bin");
    writeStatisticsMeanBinary(statistics.sumT2, statisticsSampleCount, dir / "mean_T2.bin");
    writeStatisticsMeanBinary(statistics.sumUzT, statisticsSampleCount, dir / "mean_uzT.bin");
    writeStatisticsMetadata(statisticsSampleCount, sampleStartStep, sampleEndStep, dir / "statistics_metadata.json");

    std::cout << "statistics output: " << dir << std::endl;
    std::cout << "statistics samples: " << statisticsSampleCount << std::endl;
}
