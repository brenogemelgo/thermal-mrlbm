caseName = "cavity"
runId = "000"
selectedStep = None
outputRoot = "output"
showPlots = False
figureDpi = 200

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
    readVectorComponents,
    getInt,
    getFloat,
    writeReport,
    writeDatFile,
    writeGridDatFile,
)


def saveMidplanePlots(postDir, metadata, stepValue, thetaValues, uxValues, uyValues, velocityMagnitude):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    zMid = nz // 2
    xGrid2d, yGrid2d = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")

    writeGridDatFile(
        postDir / f"cavityTemperatureMidplaneStep{stepValue}.dat",
        ["x", "y", "theta", "ux", "uy"],
        [
            xGrid2d,
            yGrid2d,
            thetaValues[zMid, :, :],
            uxValues[zMid, :, :],
            uyValues[zMid, :, :],
        ],
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(thetaValues[zMid, :, :], origin="lower", extent=(0, nx - 1, 0, ny - 1), aspect="auto")
    plt.colorbar(label="theta")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Cavity temperature, z={zMid}, step {stepValue}")

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
    plt.savefig(postDir / f"cavityTemperatureMidplaneStep{stepValue}.png", dpi=figureDpi)
    if showPlots:
        plt.show()
    plt.close()

    writeGridDatFile(
        postDir / f"cavityVelocityMidplaneStep{stepValue}.dat",
        ["x", "y", "velocityMagnitude"],
        [xGrid2d, yGrid2d, velocityMagnitude[zMid, :, :]],
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(velocityMagnitude[zMid, :, :], origin="lower", extent=(0, nx - 1, 0, ny - 1), aspect="auto")
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
    writeDatFile(
        postDir / f"cavityCenterlineProfilesStep{stepValue}.dat",
        ["x", "uyAtYMid", "y", "uxAtXMid", "thetaAtYMid"],
        [
            xValues,
            uyValues[zMid, yMid, :],
            yValues,
            uxValues[zMid, :, xMid],
            thetaValues[zMid, yMid, :],
        ],
    )

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(xValues, uyValues[zMid, yMid, :])
    plt.ylabel("uy")
    plt.title(f"Cavity centerline profiles, step {stepValue}")

    plt.subplot(3, 1, 2)
    plt.plot(yValues, uxValues[zMid, :, xMid])
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


def main():
    runDir = getRunDir(caseName, runId, outputRoot)
    metadata = readMetadata(runDir)
    postDir = getPostDir(runDir)

    temperatureValues, stepValue = readScalarField(runDir, metadata, "temperature", selectedStep)
    velocityValues, stepValue = readVectorComponents(runDir, metadata, ["ux", "uy", "uz"], stepValue)
    uxValues = velocityValues["ux"]
    uyValues = velocityValues["uy"]
    uzValues = velocityValues["uz"]

    tHot = getFloat(metadata, "THOT")
    tCold = getFloat(metadata, "TCOLD")
    deltaT = tHot - tCold
    if abs(deltaT) < 1.0e-30:
        raise RuntimeError("Cannot compute theta because THOT - TCOLD is zero")

    thetaValues = (temperatureValues - tCold) / deltaT
    lCavity = getFloat(metadata, ["L_CAVITY", "LCAVITY"])
    dThetaDxHot = (thetaValues[:, :, 1] - thetaValues[:, :, 0]) * lCavity
    nuHot = -dThetaDxHot
    nuAvg = np.mean(nuHot)
    velocityMagnitude = np.sqrt(uxValues * uxValues + uyValues * uyValues + uzValues * uzValues)
    hotWallError = np.max(np.abs(temperatureValues[:, :, 0] - tHot))
    coldWallError = np.max(np.abs(temperatureValues[:, :, -1] - tCold))

    reportLines = [
        f"selectedStep = {stepValue}",
        f"RA = {getFloat(metadata, 'RA')}",
        f"PR = {getFloat(metadata, 'PR')}",
        f"NU = {getFloat(metadata, 'NU')}",
        f"ALPHA = {getFloat(metadata, 'ALPHA')}",
        f"GBETA = {getFloat(metadata, 'GBETA')}",
        f"averageHotWallNusselt = {nuAvg:.17g}",
        f"maxVelocityMagnitude = {np.max(velocityMagnitude):.17g}",
        f"minTemperature = {np.min(temperatureValues):.17g}",
        f"maxTemperature = {np.max(temperatureValues):.17g}",
        f"hotWallMaxAbsError = {hotWallError:.17g}",
        f"coldWallMaxAbsError = {coldWallError:.17g}",
        f"hotWallMaintained = {hotWallError < 1.0e-4}",
        f"coldWallMaintained = {coldWallError < 1.0e-4}",
    ]
    writeReport(postDir / "cavityReport.txt", reportLines)

    saveMidplanePlots(postDir, metadata, stepValue, thetaValues, uxValues, uyValues, velocityMagnitude)
    saveCenterlinePlots(postDir, metadata, stepValue, thetaValues, uxValues, uyValues)


main()
