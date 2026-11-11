"""
Shared post-processing helpers for thermal-mrlbm.

Run the case scripts from the repository root:

    python3 postRayleigh.py all output/rayleigh_benard/RUN_ID

Array convention:
    HDF5 fields and legacy binaries use idx = x + y * NX + z * NX * NY.
    Helpers return arrZyx arrays with shape (NZ, NY, NX), indexed as arrZyx[z, y, x].
"""

from pathlib import Path
import json
import re

import numpy as np
from scripts.rbc_hdf5 import data_directory


momentNames = [
    "rho",
    "ux",
    "uy",
    "uz",
    "pxx",
    "pyy",
    "pzz",
    "pxy",
    "pxz",
    "pyz",
    "temperature",
    "thermalFluxX",
    "thermalFluxY",
    "thermalFluxZ",
]

fieldAliasMap = {
    "rho": "rho",
    "density": "rho",
    "ux": "ux",
    "uX": "ux",
    "velocityX": "ux",
    "uy": "uy",
    "uY": "uy",
    "velocityY": "uy",
    "uz": "uz",
    "uZ": "uz",
    "velocityZ": "uz",
    "temperature": "temperature",
    "temp": "temperature",
    "t": "temperature",
    "T": "temperature",
    "temperatureMean": "temperatureMean",
    "uzMean": "uzMean",
}

velocityFields = {"ux", "uy", "uz"}
missingValue = object()
stepPattern = re.compile(r"^step_(\d+)\.bin$")


def canonicalMetadataKey(textValue):
    return re.sub(r"[^A-Za-z0-9]", "", textValue).lower()


def parseMetadataValue(textValue):
    strippedValue = textValue.strip()
    lowerValue = strippedValue.lower()

    if lowerValue == "true":
        return True
    if lowerValue == "false":
        return False

    if re.fullmatch(r"[-+]?\d+", strippedValue):
        return int(strippedValue)

    try:
        return float(strippedValue)
    except ValueError:
        return strippedValue.strip("\"'")


def getRunDir(caseName, runId, outputRoot="output"):
    return Path(outputRoot) / caseName / runId


def readMetadata(runDir):
    if (data_directory(runDir)/'metadata.h5').exists():
        from scripts.rbc_hdf5 import read_metadata
        return read_metadata(runDir)
    jsonPath = runDir / "metadata.json"
    textPath = runDir / "metadata.txt"

    if jsonPath.exists():
        with jsonPath.open("r", encoding="utf-8") as inputFile:
            return json.load(inputFile)

    if not textPath.exists():
        raise RuntimeError(f"Missing metadata file in {runDir}")

    metadata = {}
    with textPath.open("r", encoding="utf-8") as inputFile:
        for rawLine in inputFile:
            lineText = rawLine.strip()
            if not lineText or lineText.startswith("#"):
                continue
            if "=" not in lineText:
                continue

            keyText, valueText = lineText.split("=", 1)
            keyText = keyText.strip()
            metadata[keyText] = parseMetadataValue(valueText)

    return metadata


def getMetadataValue(metadata, keyOptions, defaultValue=missingValue):
    if isinstance(keyOptions, str):
        keyOptions = [keyOptions]

    for keyText in keyOptions:
        if keyText in metadata:
            return metadata[keyText]

    canonicalOptions = {canonicalMetadataKey(keyText) for keyText in keyOptions}
    for keyText, value in metadata.items():
        if canonicalMetadataKey(keyText) in canonicalOptions:
            return value

    if defaultValue is not missingValue:
        return defaultValue

    joinedKeys = ", ".join(keyOptions)
    raise RuntimeError(f"Missing required metadata key: {joinedKeys}")


def getInt(metadata, keyOptions, defaultValue=missingValue):
    value = getMetadataValue(metadata, keyOptions, defaultValue)
    return int(value)


def getFloat(metadata, keyOptions, defaultValue=missingValue):
    value = getMetadataValue(metadata, keyOptions, defaultValue)
    return float(value)


def getBool(metadata, keyOptions, defaultValue=missingValue):
    value = getMetadataValue(metadata, keyOptions, defaultValue)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def getPostDir(runDir):
    postDir = Path(__file__).resolve().parent / 'results' / Path(runDir).name
    postDir.mkdir(parents=True, exist_ok=True)
    return postDir


def getBinaryDir(runDir):
    binaryDir = runDir / "binaries"
    if not binaryDir.exists():
        raise RuntimeError(f"Missing binary output directory: {binaryDir}")
    return binaryDir


def listAvailableSteps(runDir):
    h5steps = sorted(int(p.stem.split('_')[1]) for p in (data_directory(runDir)/'fields').glob('step_*.h5'))
    if not (Path(runDir)/'binaries').exists():
        return h5steps
    binaryDir = getBinaryDir(runDir)
    stepValues = []

    for binaryPath in binaryDir.glob("step_*.bin"):
        matchValue = stepPattern.match(binaryPath.name)
        if matchValue is not None:
            stepValues.append(int(matchValue.group(1)))

    return sorted(set(stepValues+h5steps))


def selectStep(runDir, selectedStep):
    stepValues = listAvailableSteps(runDir)
    if not stepValues:
        raise RuntimeError(f"No step binaries found in {getBinaryDir(runDir)}")

    if selectedStep is None:
        return stepValues[-1]

    selectedValue = int(selectedStep)
    if selectedValue not in stepValues:
        availableText = ", ".join(str(value) for value in stepValues)
        raise RuntimeError(f"Selected step {selectedValue} is unavailable. Available steps: {availableText}")

    return selectedValue


def getStepBinaryPath(runDir, stepValue):
    binaryPath = getBinaryDir(runDir) / f"step_{int(stepValue):09d}.bin"
    if not binaryPath.exists():
        raise RuntimeError(f"Missing step binary: {binaryPath}")
    return binaryPath


def listAvailableFields(runDir):
    paths = sorted((data_directory(runDir)/'fields').glob('step_*.h5'))
    if paths:
        import h5py
        with h5py.File(paths[-1], 'r') as source:
            return sorted(source['fields'])
    binaryDir = getBinaryDir(runDir)
    fieldNames = list(momentNames)

    for binaryPath in binaryDir.glob("*.bin"):
        if stepPattern.match(binaryPath.name) is None:
            fieldNames.append(binaryPath.stem)

    return sorted(set(fieldNames))


def canonicalFieldName(fieldName):
    if fieldName in momentNames:
        return fieldName
    if fieldName in fieldAliasMap:
        return fieldAliasMap[fieldName]

    lowerName = fieldName.lower()
    for aliasName, mappedName in fieldAliasMap.items():
        if aliasName.lower() == lowerName:
            return mappedName

    availableText = ", ".join(sorted(set(momentNames + list(fieldAliasMap.keys()))))
    raise RuntimeError(f"Unknown field alias '{fieldName}'. Available aliases: {availableText}")


def getGridShape(metadata):
    nx = getInt(metadata, "NX")
    ny = getInt(metadata, "NY")
    nz = getInt(metadata, "NZ")
    return nx, ny, nz


def getExpectedSize(metadata):
    nx, ny, nz = getGridShape(metadata)
    return nx * ny * nz


def reshapeZyx(rawValues, metadata):
    nx, ny, nz = getGridShape(metadata)
    expectedSize = nx * ny * nz

    if rawValues.size != expectedSize:
        raise RuntimeError(f"Field has {rawValues.size} values, expected {expectedSize}")

    return rawValues.reshape((nz, ny, nx))


def readPackedField(runDir, metadata, fieldName, stepValue):
    canonicalName = canonicalFieldName(fieldName)
    path = data_directory(runDir)/'fields'/f'step_{int(stepValue):09d}.h5'
    if path.exists():
        import h5py
        with h5py.File(path, 'r') as source:
            if canonicalName not in source['fields']:
                raise ValueError(f'{canonicalName} is not a physical snapshot field: {path}')
            # HDF5 velocities are already physical; never subtract acceleration twice.
            return source['fields/'+canonicalName][:]
    if canonicalName not in momentNames:
        availableText = ", ".join(listAvailableFields(runDir))
        raise RuntimeError(f"Field '{fieldName}' is not stored in step binaries. Available fields: {availableText}")

    binaryPath = getStepBinaryPath(runDir, stepValue)
    expectedSize = getExpectedSize(metadata)
    dtype = np.float64 if metadata.get("real_bytes", 4) == 8 else np.float32
    valueBytes = np.dtype(dtype).itemsize
    expectedBytes = expectedSize * len(momentNames) * valueBytes
    actualBytes = binaryPath.stat().st_size

    if actualBytes != expectedBytes:
        raise RuntimeError(f"Binary size mismatch for {binaryPath}: got {actualBytes} bytes, expected {expectedBytes}")

    fieldIndex = momentNames.index(canonicalName)
    with binaryPath.open("rb") as inputFile:
        inputFile.seek(fieldIndex * expectedSize * valueBytes)
        rawValues = np.fromfile(inputFile, dtype=dtype, count=expectedSize)

    field = reshapeZyx(rawValues, metadata)
    if canonicalName in velocityFields and metadata.get("moment_format") == "neq_post_v2":
        temperature = readPackedField(runDir, metadata, "temperature", stepValue)
        axis = {"ux": "X", "uy": "Y", "uz": "Z"}[canonicalName]
        if metadata.get("validation_case") == "V3":
            from scripts.rbc_validation_reference import exact
            component = {"X": 0, "Y": 1, "Z": 2}[axis]
            acceleration = exact("V3", getInt(metadata, "NZ"), stepValue)["a"][component][:, None, :]
        else:
            acceleration = getFloat(metadata, "GBETA") * getFloat(metadata, f"GRAVITY_{axis}") * (
                temperature - getFloat(metadata, "T_REF"))
            acceleration[[0, -1], :, :] = 0
        field = field - 0.5 * acceleration
    return field


def readScalarField(runDir, metadata, fieldName, selectedStep=None):
    stepValue = selectStep(runDir, selectedStep)
    return readPackedField(runDir, metadata, fieldName, stepValue), stepValue


def readVectorComponents(runDir, metadata, componentNames, selectedStep=None):
    stepValue = selectStep(runDir, selectedStep)
    componentValues = {}

    for componentName in componentNames:
        componentValues[componentName] = readPackedField(runDir, metadata, componentName, stepValue)

    return componentValues, stepValue


def tryReadStandaloneScalarField(runDir, metadata, fieldName):
    canonicalName = canonicalFieldName(fieldName)
    h5path = data_directory(runDir)/(canonicalName+'.h5')
    if h5path.exists():
        import h5py
        with h5py.File(h5path, 'r') as source:
            return source[canonicalName][:]
    if not (Path(runDir)/'binaries').exists():
        return None
    binaryPath = getBinaryDir(runDir) / f"{canonicalName}.bin"

    if not binaryPath.exists():
        return None

    expectedSize = getExpectedSize(metadata)
    dtype = np.float64 if metadata.get("real_bytes", 4) == 8 else np.float32
    expectedBytes = expectedSize * np.dtype(dtype).itemsize
    actualBytes = binaryPath.stat().st_size

    if actualBytes != expectedBytes:
        raise RuntimeError(f"Binary size mismatch for {binaryPath}: got {actualBytes} bytes, expected {expectedBytes}")

    rawValues = np.fromfile(binaryPath, dtype=dtype, count=expectedSize)
    return reshapeZyx(rawValues, metadata)


def makeCoordinateGrids(metadata):
    nx, ny, nz = getGridShape(metadata)
    xGrid = np.arange(nx, dtype=np.float64)[None, None, :]
    yGrid = np.arange(ny, dtype=np.float64)[None, :, None]
    zGrid = np.arange(nz, dtype=np.float64)[:, None, None]
    return xGrid, yGrid, zGrid


def writeReport(reportPath, reportLines):
    with reportPath.open("w", encoding="utf-8") as outputFile:
        for lineText in reportLines:
            outputFile.write(f"{lineText}\n")


def formatDatValue(value):
    return f"{float(value):.17g}"


def writeDatFile(datPath, columnNames, columnValues):
    preparedColumns = [np.asarray(columnValue).reshape(-1) for columnValue in columnValues]
    rowCount = max((columnValue.size for columnValue in preparedColumns), default=0)

    with datPath.open("w", encoding="utf-8") as outputFile:
        outputFile.write("# " + " ".join(columnNames) + "\n")
        for rowIndex in range(rowCount):
            rowValues = []
            for columnValue in preparedColumns:
                if rowIndex < columnValue.size:
                    rowValues.append(formatDatValue(columnValue[rowIndex]))
                else:
                    rowValues.append("nan")
            outputFile.write(" ".join(rowValues) + "\n")


def writeGridDatFile(datPath, columnNames, columnValues):
    writeDatFile(datPath, columnNames, [np.asarray(columnValue).reshape(-1) for columnValue in columnValues])
