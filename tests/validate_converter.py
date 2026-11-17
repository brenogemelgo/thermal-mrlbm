#!/usr/bin/env python3
"""CPU-only HDF5 -> VTI round trips through independent XML/Base64/VTK readers.

Requires numpy, h5py and vtk. --skip-vtk runs only XML and byte-level checks.
--large adds a bounded-memory 512^3 temperature conversion and VTK subextent
reads; no CUDA builds, solver runs or full-volume host arrays are used.
"""
import argparse
import base64
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import xml.parsers.expat

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_convert_vti import build_converter


def execute(binary, *arguments, success=True):
    result = subprocess.run([str(binary), *map(str, arguments)], text=True, capture_output=True)
    if (result.returncode == 0) != success:
        raise AssertionError(f'{result.args}\n{result.stdout}\n{result.stderr}')
    return result


def vtk_image(path, extent=None):
    from vtkmodules.vtkIOXML import vtkXMLImageDataReader
    reader = vtkXMLImageDataReader()
    errors = []
    reader.AddObserver('ErrorEvent', lambda *args: errors.append('VTK reader error'))
    reader.SetFileName(str(path))
    if extent is None:
        reader.Update()
    else:
        assert reader.UpdateExtent(extent)
    assert not errors and reader.GetErrorCode() == 0, (path, errors)
    return reader.GetOutput()


def fixture(path, run_id, step, dtype, velocity, dims=(7, 5, 11)):
    z, y, x = np.indices(dims)
    native = np.float32 if np.dtype(dtype).itemsize == 4 else np.float64
    temperature = (x + y*.125 + z*.015625).astype(native)
    # Nontrivial bit patterns check signed zero, NaN payloads, and infinities.
    temperature.flat[0] = -0.
    temperature.flat[1] = np.inf
    if native == np.float32:
        temperature.view(np.uint32).flat[2] = 0x7fc01234
    else:
        temperature.view(np.uint64).flat[2] = 0x7ff8000000001234
    values = {'temperature': temperature.astype(dtype)}
    if velocity:
        for i, name in enumerate(('ux', 'uy', 'uz')):
            values[name] = ((i+1)*x - (i+2)*y + (i+3)*z*.5).astype(dtype)
    with h5py.File(path, 'w') as f:
        f.attrs.update(step=str(step), run_id=run_id, axis_order='zyx',
                       velocity_convention='physical_u', lattice_spacing='1', wall_nodes='0,NZ-1')
        for name, data in values.items():
            f.create_dataset('fields/'+name, data=data, chunks=(1, min(3, dims[1]), min(4, dims[2])),
                             compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
    return values


def expected_arrays(values):
    scalar = values['temperature']
    dtype = scalar.dtype.newbyteorder('<')
    expected = {'temperature': scalar.astype(dtype).tobytes()}
    if 'ux' in values:
        vector = np.stack([values[k] for k in ('ux', 'uy', 'uz')], axis=-1)
        expected['velocity'] = vector.astype(dtype).tobytes()
    return expected


def check_volume(path, values, encoding, use_vtk):
    raw = path.read_bytes()
    if encoding == 'base64':
        root = ET.fromstring(raw)
        assert root.find('AppendedData').get('encoding') == 'base64'
        text = root.find('AppendedData').text.lstrip()
        assert text.startswith('_')
        payload = text[1:].rstrip().encode('ascii')
    else:
        marker = b'<AppendedData encoding="raw">\n_'
        start = raw.index(marker)+len(marker)
        root = ET.fromstring(raw[:start-len(marker)]+b'</VTKFile>')
        payload = raw[start:-len(b'\n  </AppendedData>\n</VTKFile>\n')]
    arrays = root.findall('.//PointData/DataArray')
    expected = expected_arrays(values)
    assert [array.get('Name') for array in arrays] == list(expected)
    dims = values['temperature'].shape
    extent = f'0 {dims[2]-1} 0 {dims[1]-1} 0 {dims[0]-1}'
    assert root.find('ImageData').get('WholeExtent') == extent
    assert root.find('.//Piece').get('Extent') == extent
    assert root.get('header_type') == 'UInt64'
    for i, array in enumerate(arrays):
        start = int(array.get('offset'))
        end = int(arrays[i+1].get('offset')) if i+1 < len(arrays) else len(payload)
        block = payload[start:end]
        if encoding == 'base64':
            block = base64.b64decode(block, validate=True)
        length, = struct.unpack_from('<Q', block)
        assert len(block) == 8+length
        assert block[8:] == expected[array.get('Name')], path
    if use_vtk:
        from vtkmodules.util.numpy_support import vtk_to_numpy
        image = vtk_image(path)
        assert image.GetDimensions() == dims[::-1]
        for name, reference in expected.items():
            array = image.GetPointData().GetArray(name)
            assert array is not None, name
            actual = vtk_to_numpy(array)
            assert actual.tobytes() == reference, (path, name)
        assert image.GetPointData().GetScalars().GetName() == 'temperature'
        if 'velocity' in expected:
            assert image.GetPointData().GetVectors().GetName() == 'velocity'


def check_large(binary, work, use_vtk):
    n = 512
    path = work/'large.h5'
    y, x = np.indices((n, n), dtype=np.float32)
    def reference(z):
        return (x*.001 + y*.0001 + z*.00001).astype(np.float32)
    with h5py.File(path, 'w') as f:
        f.attrs.update(step='10000', run_id='large', axis_order='zyx',
                       velocity_convention='physical_u', lattice_spacing='1', wall_nodes='0,NZ-1')
        data = f.create_dataset('fields/temperature', shape=(n, n, n), dtype='<f4',
                                chunks=(1, 64, 128), compression='gzip', compression_opts=1,
                                shuffle=True, fletcher32=True)
        for z in range(n):
            data[z] = reference(z)
    execute(binary, path, '--results-root', work/'large-results', '--jobs', 1)
    out = work/'large-results/large/vtis/step_000010000.vti'
    # Discard character data while parsing, so the large Base64 text node
    # cannot accumulate as a full-volume string in a DOM/ElementTree.
    parser = xml.parsers.expat.ParserCreate()
    with out.open('rb') as stream:
        while chunk := stream.read(65536):
            parser.Parse(chunk, False)
    parser.Parse(b'', True)
    expected_bytes = 4*((8+4*n**3+2)//3)
    assert out.stat().st_size > expected_bytes
    if use_vtk:
        from vtkmodules.util.numpy_support import vtk_to_numpy
        for extent in ((0, 3, 0, 3, 0, 2), (127, 130, 254, 257, 255, 258),
                       (508, 511, 508, 511, 509, 511)):
            image = vtk_image(out, extent)
            assert image.GetExtent() == extent
            actual = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(
                extent[5]-extent[4]+1, extent[3]-extent[2]+1, extent[1]-extent[0]+1)
            expected = np.stack([reference(z)[extent[2]:extent[3]+1, extent[0]:extent[1]+1]
                                 for z in range(extent[4], extent[5]+1)])
            assert actual.tobytes() == expected.tobytes()
    return dict(grid=n, vti_bytes=out.stat().st_size, vtk_subextent_checks=3 if use_vtk else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, help='existing converter; otherwise build only convertVTI')
    parser.add_argument('--work', type=Path, help='new or empty work directory')
    parser.add_argument('--skip-vtk', action='store_true')
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()
    if not args.skip_vtk:
        try:
            from vtkmodules.vtkCommonCore import vtkVersion
        except ImportError:
            parser.error('install vtk, or use --skip-vtk for XML/byte-level checks only')
    work = (args.work or Path(tempfile.mkdtemp(prefix='thermal-converter-validation-'))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        parser.error('--work must be new or empty')
    binary = args.binary.resolve() if args.binary else build_converter()
    source = work/'input'
    source.mkdir()
    fixtures = {}
    for i, dtype in enumerate(('<f4', '>f4', '<f8', '>f8')):
        for velocity in (False, True):
            step = 2*i+int(velocity)
            dims = (7, 5, 11) if not velocity else (3+i, 4, 7)
            path = source/f'step_{step:09d}.h5'
            fixtures[step] = fixture(path, 'roundtrip', step, dtype, velocity, dims)
    with h5py.File(source/'statistics.h5', 'w') as f:
        f.create_group('statistics')
    with h5py.File(source/'slice.h5', 'w') as f:
        f.create_dataset('fields/temperature', data=np.zeros((7, 11), dtype='f4'))
    report = dict(work=str(work), checks={}, vtk_version=None if args.skip_vtk else vtkVersion.GetVTKVersion())
    for encoding in ('base64', 'raw'):
        for jobs in (1, 4):
            results = work/f'{encoding}-jobs{jobs}'
            # Also verify that no --encoding option means Base64.
            flags = [] if encoding == 'base64' else ['--encoding', encoding]
            execute(binary, source, '--results-root', results, '--jobs', jobs, *flags)
            paths = sorted((results/'roundtrip/vtis').glob('*.vti'))
            assert len(paths) == len(fixtures)
            for path in paths:
                step = int(path.stem.removeprefix('step_'))
                check_volume(path, fixtures[step], encoding, not args.skip_vtk)
                if jobs == 4:
                    assert path.read_bytes() == (work/f'{encoding}-jobs1/roundtrip/vtis'/path.name).read_bytes()
            report['checks'][f'{encoding}-jobs{jobs}'] = f'PASS: {len(paths)} volumes; bits, extents, offsets and active arrays'

    results = work/'base64-jobs4'
    path = results/'roundtrip/vtis/step_000000000.vti'
    original = path.read_bytes()
    modified = path.stat().st_mtime_ns
    execute(binary, source, '--results-root', results, '--skip-existing')
    assert path.read_bytes() == original and path.stat().st_mtime_ns == modified
    execute(binary, source, '--results-root', results, success=False)
    assert path.read_bytes() == original
    for bad in (original.replace(b'</VTKFile>', b'</VTKFilX>'),
                original.replace(b'Float32', b'Float64', 1), original[:len(original)//2]):
        path.write_bytes(bad)
        result = execute(binary, source/'step_000000000.h5', '--results-root', results,
                         '--skip-existing', success=False)
        assert 'incompatible or incomplete' in result.stderr
        assert path.read_bytes() == bad
        execute(binary, source/'step_000000000.h5', '--results-root', results, '--overwrite')
        assert path.read_bytes() == original
    execute(binary, source/'step_000000000.h5', '--results-root', results,
            '--encoding', 'raw', '--skip-existing', success=False)
    execute(binary, source, '--results-root', results, '--skip-existing', '--overwrite', success=False)
    execute(binary, source, '--encoding', 'invalid', success=False)
    report['checks']['existing-outputs'] = 'PASS: no silent overwrites; skip checks header, footer, size and encoding; explicit repair'

    # Inject a checksum-invalid HDF5 chunk. Failure during overwrite must keep
    # the previous published VTI and remove the incomplete temporary file.
    with h5py.File(source/'step_000000000.h5', 'r+') as f:
        f['fields/temperature'].id.write_direct_chunk((0, 0, 0), b'broken chunk', filter_mask=0)
    execute(binary, source/'step_000000000.h5', '--results-root', results, '--overwrite', success=False)
    assert path.read_bytes() == original
    assert not path.with_suffix('.vti.tmp').exists()
    report['checks']['failed-overwrite'] = 'PASS: published output preserved after HDF5 read failure'
    if args.large:
        report['checks']['large'] = check_large(binary, work, not args.skip_vtk)
    (work/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
