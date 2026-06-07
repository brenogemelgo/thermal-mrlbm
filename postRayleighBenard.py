caseName = "rayleigh_benard"
runId = "000"
selectedStep = None
outputRoot = "output"
showPlots = False
figureDpi = 600

import numpy as np
import matplotlib

if not showPlots:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from postCommon import (
    getRunDir,
    readMetadata,
    getPostDir,
    readScalarField,
    getInt,
    getFloat,
    getBool,
    writeReport,
    writeDatFile,
    writeGridDatFile,
)


def tryReadField(runDir, metadata, fieldName, stepValue):
    try:
        fieldValues, _ = readScalarField(runDir, metadata, fieldName, stepValue)
        return fieldValues
    except RuntimeError:
        return None


def requireExpectedShape(fieldValues, metadata, fieldName):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")

    expectedShape = (nz, ny, nx)

    if fieldValues.shape != expectedShape:
        raise RuntimeError(
            f"{fieldName} has shape {fieldValues.shape}, "
            f"but expected {expectedShape} = (NZ, NY, NX). "
            "This post-processing script assumes fields are indexed as [z, y, x]."
        )


def conductiveProfile(metadata):
    nz = getInt(metadata, "NZ")
    hValue = getFloat(metadata, "H")
    tHot = getFloat(metadata, "THOT")
    deltaT = getFloat(metadata, "DELTA_T")

    zIndex = np.arange(nz, dtype=np.float64)
    zCoord = zIndex / hValue

    temperatureExact = tHot - deltaT * zCoord

    return zIndex, zCoord, temperatureExact


def wallDerivativeBottom(meanTemperature):
    if meanTemperature.size >= 3:
        return 0.5 * (
            -3.0 * meanTemperature[0] + 4.0 * meanTemperature[1] - meanTemperature[2]
        )

    return meanTemperature[1] - meanTemperature[0]


def wallDerivativeTop(meanTemperature):
    if meanTemperature.size >= 3:
        return 0.5 * (
            3.0 * meanTemperature[-1] - 4.0 * meanTemperature[-2] + meanTemperature[-3]
        )

    return meanTemperature[-1] - meanTemperature[-2]


def safeDivide(numerator, denominator):
    if not np.isfinite(denominator):
        return np.nan

    if abs(denominator) <= 1.0e-30:
        return np.nan

    return numerator / denominator


def computeVelocityMetrics(metadata, uxValues, uyValues, uzValues):
    if uxValues is None or uyValues is None or uzValues is None:
        return {
            "velocityMax": np.nan,
            "velocityRms": np.nan,
            "kineticEnergy": np.nan,
            "velocityMaxOverFreefall": np.nan,
            "velocityRmsOverFreefall": np.nan,
        }

    velocityMagnitudeSquared = (
        uxValues * uxValues + uyValues * uyValues + uzValues * uzValues
    )

    velocityMagnitude = np.sqrt(velocityMagnitudeSquared)

    velocityMax = np.max(velocityMagnitude)
    velocityRms = np.sqrt(np.mean(velocityMagnitudeSquared))
    kineticEnergy = 0.5 * np.mean(velocityMagnitudeSquared)

    uFreefall = getFloat(metadata, "U_FREEFALL", 0.0)

    return {
        "velocityMax": velocityMax,
        "velocityRms": velocityRms,
        "kineticEnergy": kineticEnergy,
        "velocityMaxOverFreefall": safeDivide(velocityMax, uFreefall),
        "velocityRmsOverFreefall": safeDivide(velocityRms, uFreefall),
    }


def computeNusseltMetrics(metadata, meanTemperature, temperatureError, uzValues):
    hValue = getFloat(metadata, "H")
    deltaT = getFloat(metadata, "DELTA_T")
    alphaValue = getFloat(metadata, "ALPHA")

    dTdzBottom = wallDerivativeBottom(meanTemperature)
    dTdzTop = wallDerivativeTop(meanTemperature)

    nuBottom = -hValue / deltaT * dTdzBottom
    nuTop = -hValue / deltaT * dTdzTop
    nuAverage = 0.5 * (nuBottom + nuTop)

    if uzValues is None:
        nuVolume = np.nan
    else:
        # Lattice-unit form:
        #
        # Nu = 1 + H / (alpha DeltaT) <u_z theta>
        #
        # where theta = T - T_conductive.
        #
        # This is equivalent to the usual nondimensional
        # Nu = 1 + sqrt(Ra Pr) <w theta>
        # only after free-fall nondimensionalization.
        nuVolume = 1.0 + hValue / (alphaValue * deltaT) * np.mean(
            uzValues * temperatureError
        )

    return {
        "dTdzBottom": dTdzBottom,
        "dTdzTop": dTdzTop,
        "nuBottom": nuBottom,
        "nuTop": nuTop,
        "nuAverage": nuAverage,
        "nuVolume": nuVolume,
    }


def saveProfilePlot(postDir, stepValue, zCoord, meanTemperature, temperatureExact):
    writeDatFile(
        postDir / f"rayleighBenardMeanTemperatureStep{stepValue}.dat",
        ["Z", "meanTemperature", "conductiveTemperature", "temperatureDifference"],
        [zCoord, meanTemperature, temperatureExact, meanTemperature - temperatureExact],
    )

    plt.figure(figsize=(5, 6))
    plt.plot(meanTemperature, zCoord, label="simulation")
    plt.plot(temperatureExact, zCoord, "--", label="conduction")
    plt.xlabel("temperature")
    plt.ylabel("Z")
    plt.title(f"Mean temperature profile, step {stepValue}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        postDir / f"rayleighBenardMeanTemperatureStep{stepValue}.png",
        dpi=figureDpi,
    )

    if showPlots:
        plt.show()

    plt.close()


def saveSlicePlots(
    postDir,
    metadata,
    stepValue,
    temperatureValues,
    temperatureError,
    uxValues,
    uzValues,
):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")

    yMid = ny // 2

    xGrid2d, zGrid2d = np.meshgrid(
        np.arange(nx),
        np.arange(nz),
        indexing="xy",
    )

    temperatureSlice = temperatureValues[:, yMid, :]
    temperatureErrorSlice = temperatureError[:, yMid, :]

    writeGridDatFile(
        postDir / f"rayleighBenardTemperatureXzStep{stepValue}.dat",
        ["x", "z", "temperature"],
        [xGrid2d, zGrid2d, temperatureSlice],
    )

    plt.figure(figsize=(8, 4))
    plt.imshow(
        temperatureSlice,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="auto",
    )
    plt.colorbar(label="temperature")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Rayleigh-Benard temperature, y={yMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(
        postDir / f"rayleighBenardTemperatureXzStep{stepValue}.png",
        dpi=figureDpi,
    )

    if showPlots:
        plt.show()

    plt.close()

    writeGridDatFile(
        postDir / f"rayleighBenardTemperatureErrorXzStep{stepValue}.dat",
        ["x", "z", "temperatureError"],
        [xGrid2d, zGrid2d, temperatureErrorSlice],
    )

    plt.figure(figsize=(8, 4))
    plt.imshow(
        temperatureErrorSlice,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="auto",
    )
    plt.colorbar(label="temperature error")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Rayleigh-Benard T - T_conduction, y={yMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(
        postDir / f"rayleighBenardTemperatureErrorXzStep{stepValue}.png",
        dpi=figureDpi,
    )

    if showPlots:
        plt.show()

    plt.close()

    if uzValues is None:
        return

    uzSlice = uzValues[:, yMid, :]

    writeGridDatFile(
        postDir / f"rayleighBenardUzXzStep{stepValue}.dat",
        ["x", "z", "uz"],
        [xGrid2d, zGrid2d, uzSlice],
    )

    plt.figure(figsize=(8, 4))
    plt.imshow(
        uzSlice,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="auto",
    )
    plt.colorbar(label="uz")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Rayleigh-Benard uz, y={yMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(
        postDir / f"rayleighBenardUzXzStep{stepValue}.png",
        dpi=figureDpi,
    )

    if showPlots:
        plt.show()

    plt.close()

    if uxValues is None:
        return

    uxSlice = uxValues[:, yMid, :]

    dUxDz = np.gradient(uxSlice, axis=0)
    dUzDx = np.gradient(uzSlice, axis=1)

    omegaY = dUxDz - dUzDx

    writeGridDatFile(
        postDir / f"rayleighBenardOmegaYStep{stepValue}.dat",
        ["x", "z", "omegaY"],
        [xGrid2d, zGrid2d, omegaY],
    )

    plt.figure(figsize=(8, 4))
    plt.imshow(
        omegaY,
        origin="lower",
        extent=(0, nx - 1, 0, nz - 1),
        aspect="auto",
        cmap="coolwarm",
    )
    plt.colorbar(label="omega_y")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Rayleigh-Benard omega_y, y={yMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(
        postDir / f"rayleighBenardOmegaYStep{stepValue}.png",
        dpi=figureDpi,
    )

    if showPlots:
        plt.show()

    plt.close()


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    postDir = getPostDir(runDir)

    temperatureValues, stepValue = readScalarField(
        runDir,
        metadata,
        "temperature",
        selectedStep,
    )

    requireExpectedShape(temperatureValues, metadata, "temperature")

    uxValues = tryReadField(runDir, metadata, "ux", stepValue)
    uyValues = tryReadField(runDir, metadata, "uy", stepValue)
    uzValues = tryReadField(runDir, metadata, "uz", stepValue)
    rhoValues = tryReadField(runDir, metadata, "rho", stepValue)

    for fieldName, fieldValues in [
        ("ux", uxValues),
        ("uy", uyValues),
        ("uz", uzValues),
        ("rho", rhoValues),
    ]:
        if fieldValues is not None:
            requireExpectedShape(fieldValues, metadata, fieldName)

    zIndex, zCoord, temperatureExact = conductiveProfile(metadata)

    meanTemperature = np.mean(temperatureValues, axis=(1, 2))
    conductiveField = temperatureExact[:, None, None]

    temperatureError = temperatureValues - conductiveField
    meanTemperatureError = meanTemperature - temperatureExact

    temperatureL2Error = np.sqrt(np.mean(temperatureError * temperatureError))
    temperatureLinfError = np.max(np.abs(temperatureError))

    meanTemperatureL2Error = np.sqrt(
        np.mean(meanTemperatureError * meanTemperatureError)
    )
    meanTemperatureLinfError = np.max(np.abs(meanTemperatureError))

    horizontalTemperatureStd = np.std(temperatureValues, axis=(1, 2))
    horizontalTemperatureStdMax = np.max(horizontalTemperatureStd)
    horizontalTemperatureStdMean = np.mean(horizontalTemperatureStd)

    tHot = getFloat(metadata, "THOT")
    tCold = getFloat(metadata, "TCOLD")

    wallBottomTemperatureError = np.max(np.abs(temperatureValues[0, :, :] - tHot))
    wallTopTemperatureError = np.max(np.abs(temperatureValues[-1, :, :] - tCold))

    velocityMetrics = computeVelocityMetrics(
        metadata,
        uxValues,
        uyValues,
        uzValues,
    )

    nusseltMetrics = computeNusseltMetrics(
        metadata,
        meanTemperature,
        temperatureError,
        uzValues,
    )

    reportLines = [
        f"selectedStep = {stepValue}",
        f"RA = {getFloat(metadata, 'RA')}",
        f"PR = {getFloat(metadata, 'PR')}",
        f"NU = {getFloat(metadata, 'NU')}",
        f"ALPHA = {getFloat(metadata, 'ALPHA')}",
        f"GBETA = {getFloat(metadata, 'GBETA')}",
        f"U_FREEFALL = {getFloat(metadata, 'U_FREEFALL', 0.0)}",
        f"U_CHAR = {getFloat(metadata, 'U_CHAR', 0.0)}",
        f"RE = {getFloat(metadata, 'RE', 0.0)}",
        f"H = {getFloat(metadata, 'H')}",
        f"ASPECT_RATIO = {getFloat(metadata, 'ASPECT_RATIO')}",
        f"CONDUCTION_ONLY = {getBool(metadata, 'CONDUCTION_ONLY', False)}",
        f"PERTURBATION_AMPLITUDE = {getFloat(metadata, 'PERTURBATION_AMPLITUDE', 0.0)}",
        f"temperatureL2Error = {temperatureL2Error:.17g}",
        f"temperatureLinfError = {temperatureLinfError:.17g}",
        f"meanTemperatureL2Error = {meanTemperatureL2Error:.17g}",
        f"meanTemperatureLinfError = {meanTemperatureLinfError:.17g}",
        f"horizontalTemperatureStdMax = {horizontalTemperatureStdMax:.17g}",
        f"horizontalTemperatureStdMean = {horizontalTemperatureStdMean:.17g}",
        f"wallBottomTemperatureError = {wallBottomTemperatureError:.17g}",
        f"wallTopTemperatureError = {wallTopTemperatureError:.17g}",
        f"dTdzBottom = {nusseltMetrics['dTdzBottom']:.17g}",
        f"dTdzTop = {nusseltMetrics['dTdzTop']:.17g}",
        f"nuBottom = {nusseltMetrics['nuBottom']:.17g}",
        f"nuTop = {nusseltMetrics['nuTop']:.17g}",
        f"nuAverage = {nusseltMetrics['nuAverage']:.17g}",
        f"nuVolume = {nusseltMetrics['nuVolume']:.17g}",
        "nuVolume uses lattice-unit form: 1 + H/(ALPHA*DELTA_T)*mean(uz*(T-T_conduction)).",
        f"velocityMax = {velocityMetrics['velocityMax']:.17g}",
        f"velocityRms = {velocityMetrics['velocityRms']:.17g}",
        f"kineticEnergy = {velocityMetrics['kineticEnergy']:.17g}",
        f"velocityMaxOverFreefall = {velocityMetrics['velocityMaxOverFreefall']:.17g}",
        f"velocityRmsOverFreefall = {velocityMetrics['velocityRmsOverFreefall']:.17g}",
        "free-fall normalized velocity metrics are NaN when U_FREEFALL = 0, as in pure conduction.",
        f"temperatureMin = {np.min(temperatureValues):.17g}",
        f"temperatureMax = {np.max(temperatureValues):.17g}",
    ]

    if rhoValues is not None:
        reportLines += [
            f"rhoMin = {np.min(rhoValues):.17g}",
            f"rhoMax = {np.max(rhoValues):.17g}",
            f"rhoMean = {np.mean(rhoValues):.17g}",
            f"rhoStd = {np.std(rhoValues):.17g}",
        ]
    else:
        reportLines += [
            "rhoMin = unavailable",
            "rhoMax = unavailable",
            "rhoMean = unavailable",
            "rhoStd = unavailable",
        ]

    writeReport(postDir / "rayleighBenardReport.txt", reportLines)

    for lineText in reportLines:
        print(lineText)

    saveProfilePlot(
        postDir,
        stepValue,
        zCoord,
        meanTemperature,
        temperatureExact,
    )

    saveSlicePlots(
        postDir,
        metadata,
        stepValue,
        temperatureValues,
        temperatureError,
        uxValues,
        uzValues,
    )

    writeDatFile(
        postDir / f"rayleighBenardHorizontalTemperatureStdStep{stepValue}.dat",
        ["Z", "horizontalTemperatureStd"],
        [zCoord, horizontalTemperatureStd],
    )

    writeDatFile(
        postDir / f"rayleighBenardSummaryStep{stepValue}.dat",
        [
            "step",
            "temperatureL2Error",
            "temperatureLinfError",
            "meanTemperatureL2Error",
            "meanTemperatureLinfError",
            "horizontalTemperatureStdMax",
            "wallBottomTemperatureError",
            "wallTopTemperatureError",
            "velocityMax",
            "velocityRms",
            "kineticEnergy",
            "velocityMaxOverFreefall",
            "velocityRmsOverFreefall",
            "nuBottom",
            "nuTop",
            "nuAverage",
            "nuVolume",
        ],
        [
            np.array([stepValue]),
            np.array([temperatureL2Error]),
            np.array([temperatureLinfError]),
            np.array([meanTemperatureL2Error]),
            np.array([meanTemperatureLinfError]),
            np.array([horizontalTemperatureStdMax]),
            np.array([wallBottomTemperatureError]),
            np.array([wallTopTemperatureError]),
            np.array([velocityMetrics["velocityMax"]]),
            np.array([velocityMetrics["velocityRms"]]),
            np.array([velocityMetrics["kineticEnergy"]]),
            np.array([velocityMetrics["velocityMaxOverFreefall"]]),
            np.array([velocityMetrics["velocityRmsOverFreefall"]]),
            np.array([nusseltMetrics["nuBottom"]]),
            np.array([nusseltMetrics["nuTop"]]),
            np.array([nusseltMetrics["nuAverage"]]),
            np.array([nusseltMetrics["nuVolume"]]),
        ],
    )


main()
