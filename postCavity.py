caseName = "cavity"
runId = "1e5"
selectedStep = None
outputRoot = "output"
showPlots = False
figureDpi = 600
benchmarkTolerance = 0.01

# Standard average-Nusselt benchmark values for the de Vahl Davis natural-convection
# square cavity; small variations exist depending on exact Nusselt definition and grid.
deVahlDavisNuAvg = {
    1.0e3: 1.118,
    1.0e4: 2.244,
    1.0e5: 4.52,
    1.0e6: 8.90,
}

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
    writeReport,
    writeDatFile,
    writeGridDatFile,
)


def saveMidplanePlots(
    postDir, metadata, stepValue, thetaValues, uxValues, uyValues, velocityMagnitude
):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    zMid = nz // 2
    xGrid2d, yGrid2d = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    uxMidplane = (
        uxValues[zMid, :, :] if uxValues is not None else np.full((ny, nx), np.nan)
    )
    uyMidplane = (
        uyValues[zMid, :, :] if uyValues is not None else np.full((ny, nx), np.nan)
    )

    writeGridDatFile(
        postDir / f"cavityTemperatureMidplaneStep{stepValue}.dat",
        ["x", "y", "theta", "ux", "uy"],
        [
            xGrid2d,
            yGrid2d,
            thetaValues[zMid, :, :],
            uxMidplane,
            uyMidplane,
        ],
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(
        thetaValues[zMid, :, :],
        origin="lower",
        extent=(0, nx - 1, 0, ny - 1),
        aspect="auto",
    )
    plt.colorbar(label="theta")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Cavity temperature, z={zMid}, step {stepValue}")

    if uxValues is not None and uyValues is not None:
        strideValue = max(1, min(nx, ny) // 24)
        xValues = np.arange(nx)[::strideValue]
        yValues = np.arange(ny)[::strideValue]
        plt.quiver(
            xValues,
            yValues,
            uxValues[zMid, ::strideValue, ::strideValue],
            uyValues[zMid, ::strideValue, ::strideValue],
            color="white",
            scale=None,
            width=0.002,
        )
    plt.tight_layout()
    plt.savefig(
        postDir / f"cavityTemperatureMidplaneStep{stepValue}.png", dpi=figureDpi
    )
    if showPlots:
        plt.show()
    plt.close()

    if velocityMagnitude is None:
        return

    writeGridDatFile(
        postDir / f"cavityVelocityMidplaneStep{stepValue}.dat",
        ["x", "y", "velocityMagnitude"],
        [xGrid2d, yGrid2d, velocityMagnitude[zMid, :, :]],
    )
    plt.figure(figsize=(6, 5))
    plt.imshow(
        velocityMagnitude[zMid, :, :],
        origin="lower",
        extent=(0, nx - 1, 0, ny - 1),
        aspect="auto",
    )
    plt.colorbar(label="|u|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Cavity velocity magnitude, z={zMid}, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"cavityVelocityMidplaneStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def saveCenterlinePlots(postDir, metadata, stepValue, thetaValues, uxValues, uyValues):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    zMid = nz // 2
    xMid = nx // 2
    yMid = ny // 2

    xValues = np.arange(nx)
    yValues = np.arange(ny)
    lCavity = getFloat(metadata, ["L_CAVITY", "LCAVITY"])
    thetaHorizontal = thetaValues[zMid, yMid, :]
    thetaVertical = thetaValues[zMid, :, xMid]
    uyHorizontal = (
        uyValues[zMid, yMid, :] if uyValues is not None else np.full(nx, np.nan)
    )
    uxVertical = (
        uxValues[zMid, :, xMid] if uxValues is not None else np.full(ny, np.nan)
    )

    centerlineNames = ["x", "y", "thetaAtYMid"]
    centerlineValues = [xValues, yValues, thetaHorizontal]
    horizontalNames = ["xIndex", "X", "theta"]
    horizontalValues = [xValues, xValues / lCavity, thetaHorizontal]
    verticalNames = ["yIndex", "Y", "theta"]
    verticalValues = [yValues, yValues / lCavity, thetaVertical]

    if uyValues is not None:
        centerlineNames.append("uyAtYMid")
        centerlineValues.append(uyHorizontal)
        horizontalNames.append("uy")
        horizontalValues.append(uyHorizontal)

    if uxValues is not None:
        centerlineNames.append("uxAtXMid")
        centerlineValues.append(uxVertical)
        verticalNames.append("ux")
        verticalValues.append(uxVertical)

    writeDatFile(
        postDir / f"cavityCenterlineProfilesStep{stepValue}.dat",
        centerlineNames,
        centerlineValues,
    )
    writeDatFile(
        postDir / f"cavityHorizontalCenterlineStep{stepValue}.txt",
        horizontalNames,
        horizontalValues,
    )
    writeDatFile(
        postDir / f"cavityVerticalCenterlineStep{stepValue}.txt",
        verticalNames,
        verticalValues,
    )

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(xValues, uyHorizontal)
    plt.ylabel("uy")
    plt.title(f"Cavity centerline profiles, step {stepValue}")

    plt.subplot(3, 1, 2)
    plt.plot(yValues, uxVertical)
    plt.ylabel("ux")

    plt.subplot(3, 1, 3)
    plt.plot(xValues, thetaValues[zMid, yMid, :])
    plt.xlabel("coordinate")
    plt.ylabel("theta")

    plt.tight_layout()
    plt.savefig(postDir / f"cavityCenterlineProfilesStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def findClosestBenchmarkRa(raValue, benchmarkTable, benchmarkTolerance):
    for benchmarkRa in benchmarkTable:
        relativeDifference = abs(raValue - benchmarkRa) / benchmarkRa
        if relativeDifference <= benchmarkTolerance:
            return benchmarkRa
    return None


def saveNusseltPlots(
    postDir, metadata, stepValue, nuAvgHot, matchedRa, nuBenchmark, nuHotForSpread
):
    ny = getInt(metadata, "NY")
    lCavity = getFloat(metadata, ["L_CAVITY", "LCAVITY"])

    benchmarkValues = np.array(
        [nuAvgHot, nuBenchmark if matchedRa is not None else np.nan]
    )
    writeDatFile(
        postDir / f"cavityNusseltBenchmarkStep{stepValue}.dat",
        ["index", "nuValue"],
        [np.array([0, 1]), benchmarkValues],
    )

    plt.figure(figsize=(5, 4))
    if matchedRa is not None:
        nuErrorPercent = 100.0 * (nuAvgHot - nuBenchmark) / nuBenchmark
        plt.bar([0, 1], benchmarkValues, tick_label=["simulation", "benchmark"])
        plt.title(f"RA={matchedRa:.0e}, error={nuErrorPercent:.2f}%")
    else:
        plt.bar([0], [nuAvgHot], tick_label=["simulation"])
        plt.title("No configured benchmark RA")
    plt.ylabel("average hot-wall Nu")
    plt.tight_layout()
    plt.savefig(postDir / f"cavityNusseltBenchmarkStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    nuHotMeanY = np.mean(nuHotForSpread, axis=0)
    yCoord = np.arange(ny) / lCavity
    writeDatFile(
        postDir / f"cavityHotWallNusseltStep{stepValue}.dat",
        ["Y", "nuHotMeanY"],
        [yCoord, nuHotMeanY],
    )
    plt.figure(figsize=(7, 4))
    plt.plot(yCoord, nuHotMeanY)
    plt.xlabel("Y")
    plt.ylabel("hot-wall Nu")
    plt.title(f"Hot-wall Nusselt distribution, step {stepValue}")
    plt.tight_layout()
    plt.savefig(postDir / f"cavityHotWallNusseltStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()


def tryReadScalarField(runDir, metadata, fieldName, stepValue):
    try:
        fieldValues, _ = readScalarField(runDir, metadata, fieldName, stepValue)
        return fieldValues
    except RuntimeError:
        return None


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    postDir = getPostDir(runDir)

    temperatureValues, stepValue = readScalarField(
        runDir, metadata, "temperature", selectedStep
    )
    uxValues = tryReadScalarField(runDir, metadata, "ux", stepValue)
    uyValues = tryReadScalarField(runDir, metadata, "uy", stepValue)
    uzValues = tryReadScalarField(runDir, metadata, "uz", stepValue)

    tHot = getFloat(metadata, "THOT")
    tCold = getFloat(metadata, "TCOLD")
    deltaT = tHot - tCold
    if abs(deltaT) < 1.0e-30:
        raise RuntimeError("Cannot compute theta because THOT - TCOLD is zero")

    thetaValues = (temperatureValues - tCold) / deltaT
    lCavity = getFloat(metadata, ["L_CAVITY", "LCAVITY"])
    dThetaDxHotFirstOrder = (thetaValues[:, :, 1] - thetaValues[:, :, 0]) * lCavity
    nuHotFirstOrder = -dThetaDxHotFirstOrder
    nuAvgHotFirstOrder = np.mean(nuHotFirstOrder)

    nx = getInt(metadata, "NX")
    if nx >= 3:
        dThetaDxHotSecondOrder = (
            (
                -3.0 * thetaValues[:, :, 0]
                + 4.0 * thetaValues[:, :, 1]
                - thetaValues[:, :, 2]
            )
            * 0.5
            * lCavity
        )
        nuHotSecondOrder = -dThetaDxHotSecondOrder
        nuAvgHotSecondOrder = np.mean(nuHotSecondOrder)
        nuHotForSpread = nuHotSecondOrder
        nuAvgHot = nuAvgHotSecondOrder
    else:
        nuHotSecondOrder = None
        nuAvgHotSecondOrder = np.nan
        nuHotForSpread = nuHotFirstOrder
        nuAvgHot = nuAvgHotFirstOrder

    if uxValues is not None and uyValues is not None and uzValues is not None:
        velocityMagnitude = np.sqrt(
            uxValues * uxValues + uyValues * uyValues + uzValues * uzValues
        )
        velocityMax = np.max(velocityMagnitude)
    elif uxValues is not None and uyValues is not None:
        velocityMagnitude = np.sqrt(uxValues * uxValues + uyValues * uyValues)
        velocityMax = np.max(velocityMagnitude)
    else:
        velocityMagnitude = None
        velocityMax = None

    raValue = getFloat(metadata, "RA")
    matchedRa = findClosestBenchmarkRa(raValue, deVahlDavisNuAvg, benchmarkTolerance)
    nuBenchmark = deVahlDavisNuAvg[matchedRa] if matchedRa is not None else np.nan
    nuError = nuAvgHot - nuBenchmark if matchedRa is not None else np.nan
    nuErrorPercent = 100.0 * nuError / nuBenchmark if matchedRa is not None else np.nan
    if matchedRa is None:
        quality = "notAvailable"
    elif abs(nuErrorPercent) < 2.0:
        quality = "excellent"
    elif abs(nuErrorPercent) < 5.0:
        quality = "acceptable"
    else:
        quality = "needsReview"

    thetaMeanZ = np.mean(thetaValues, axis=0)
    thetaStdZ = np.std(thetaValues, axis=0)
    spanwiseRelativeStd = np.max(thetaStdZ) / max(np.max(np.abs(thetaMeanZ)), 1.0e-30)
    nuHotZ = np.mean(nuHotForSpread, axis=1)
    nuHotZMin = np.min(nuHotZ)
    nuHotZMax = np.max(nuHotZ)
    nuHotZSpreadPercent = 100.0 * (nuHotZMax - nuHotZMin) / max(abs(nuAvgHot), 1.0e-30)
    hotWallMeanTemperature = np.mean(temperatureValues[:, :, 0])
    coldWallMeanTemperature = np.mean(temperatureValues[:, :, -1])
    hotWallError = hotWallMeanTemperature - tHot
    coldWallError = coldWallMeanTemperature - tCold

    reportLines = [
        f"selectedStep = {stepValue}",
        f"RA = {raValue}",
        f"PR = {getFloat(metadata, 'PR')}",
        f"NU = {getFloat(metadata, 'NU')}",
        f"ALPHA = {getFloat(metadata, 'ALPHA')}",
        f"GBETA = {getFloat(metadata, 'GBETA')}",
        f"L_CAVITY = {lCavity}",
        f"THOT = {tHot}",
        f"TCOLD = {tCold}",
        f"T_REF = {getFloat(metadata, 'T_REF')}",
        f"nuAvgHotFirstOrder = {nuAvgHotFirstOrder:.17g}",
        f"nuAvgHotSecondOrder = {nuAvgHotSecondOrder:.17g}",
        f"nuAvgHotReported = {nuAvgHot:.17g}",
        f"benchmarkRa = {matchedRa if matchedRa is not None else 'none'}",
        f"nuBenchmark = {nuBenchmark:.17g}",
        f"nuError = {nuError:.17g}",
        f"nuErrorPercent = {nuErrorPercent:.17g}",
        f"quality = {quality}",
        f"spanwiseRelativeStd = {spanwiseRelativeStd:.17g}",
        f"nuHotZMin = {nuHotZMin:.17g}",
        f"nuHotZMax = {nuHotZMax:.17g}",
        f"nuHotZSpreadPercent = {nuHotZSpreadPercent:.17g}",
        f"temperatureMin = {np.min(temperatureValues):.17g}",
        f"temperatureMax = {np.max(temperatureValues):.17g}",
        f"hotWallMeanTemperature = {hotWallMeanTemperature:.17g}",
        f"coldWallMeanTemperature = {coldWallMeanTemperature:.17g}",
        f"hotWallError = {hotWallError:.17g}",
        f"coldWallError = {coldWallError:.17g}",
        (
            f"velocityMax = {velocityMax:.17g}"
            if velocityMax is not None
            else "velocityMax = unavailable"
        ),
        "Benchmark comparison assumes quasi-2D behavior. Check spanwiseRelativeStd and nuHotZSpreadPercent.",
    ]
    if matchedRa is None:
        reportLines.append(
            "No configured de Vahl Davis benchmark for RA = "
            + str(raValue)
            + ". Available values: 1e3, 1e4, 1e5, 1e6."
        )
    writeReport(postDir / "cavityReport.txt", reportLines)

    saveMidplanePlots(
        postDir, metadata, stepValue, thetaValues, uxValues, uyValues, velocityMagnitude
    )
    saveCenterlinePlots(postDir, metadata, stepValue, thetaValues, uxValues, uyValues)
    saveNusseltPlots(
        postDir, metadata, stepValue, nuAvgHot, matchedRa, nuBenchmark, nuHotForSpread
    )


main()
